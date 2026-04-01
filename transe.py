import torch
import torch.nn as nn
import logging
import os
from typing import Tuple
from torch.optim import Adam, SGD, Adagrad, RMSprop

import config
from base_model import BaseModule, BaseModel
from datasets import batch_by_num

OPTIMIZER_MAP = {
    'Adam': Adam,
    'SGD': SGD,
    'Adagrad': Adagrad,
    'RMSprop': RMSprop,
}

EPSILON = 1e-8

class TransEModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'TransE'

        self.dim = config.dim
        self.margin = config.margin
        self.p = config.p
        self.temp = config.temp

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.normal_(0, 1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.entity_embed(tail) - self.entity_embed(head) - self.relation_embed(relation) + EPSILON, p=self.p, dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation ,tail) / self.temp

    def constraint(self) -> None:
        self.entity_embed.weight.data.renorm_(2, 0, 1)
        self.relation_embed.weight.data.renorm_(2, 0, 1)

class TransE(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'TransE'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate

        self.model = TransEModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.margin = self.model.margin
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> tuple[float, int]:
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        best_epoch = -1
        patience_counter = 0
        for epoch in range(self.n_epoch):
            epoch_loss = 0
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx]
            relation = relation[rand_idx]
            tail = tail[rand_idx]
            head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
            head_device = head.to(config.device)
            relation_device = relation.to(config.device)
            tail_device = tail.to(config.device)
            head_corrupted = head_corrupted.to(config.device)
            tail_corrupted = tail_corrupted.to(config.device)
            
            for h0, r, t0, h1, t1 in batch_by_num(self.n_batch, head_device, relation_device, tail_device,
                                                  head_corrupted, tail_corrupted, n_sample=n_train):
                self.model.zero_grad()
                loss = torch.sum(self.model.pair_loss(h0, r, t0, h1, t1))
                loss.backward()
                self.opt.step()
                self.model.constraint()
                epoch_loss += loss.item()
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()
                if (test_perf > best_perf):
                    self.save()
                    best_perf = test_perf
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, early_stop_patience)
                    break
        self.load(self.model_path)
        return best_perf, best_epoch