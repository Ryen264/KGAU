import torch
import torch.nn as nn
import logging
import os
from typing import Tuple
from torch.optim import Adam, SGD, Adagrad, RMSprop

import config
from base_model import BaseModule, BaseModel

OPTIMIZER_MAP = {
    'Adam': Adam,
    'SGD': SGD,
    'Adagrad': Adagrad,
    'RMSprop': RMSprop,
}

EPSILON = 1e-8

class DirectAU_KGModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'DirectAU_KG'

        self.dim = config.dim
        self.gamma = getattr(config, 'gamma', 0.1)
        self.compose_mode = getattr(config, 'compose_mode', 'mul') # 'mul' (Hadamard) or 'add'

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.relation_mask = nn.Embedding(self.n_relation, self.dim)
        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        self.entity_embed.weight.data.normal_(0, 1 / self.dim ** 0.5)
        self.relation_embed.weight.data.normal_(0, 1 / self.dim ** 0.5)
        self.relation_mask.weight.data.normal_(0, 1 / self.dim ** 0.5)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Projects vectors onto the unit hypersphere."""
        return x / (x.norm(p=2, dim=-1, keepdim=True) + EPSILON)

    def _compose(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Composes head and relation, then re-normalizes."""
        if self.compose_mode == 'mul':
            q_raw = h * r
        else:
            q_raw = h + r
        return self._normalize(q_raw)

    def _mask_head(self, h: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.relation_mask(relation))
        return self._normalize(h * attn)

    def batch_uniformity_loss(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        batch_entities = torch.unique(torch.cat([head, tail], dim=0))
        if batch_entities.numel() < 2:
            return torch.zeros((), device=head.device)

        e_batch = self._normalize(self.entity_embed(batch_entities))
        pairwise_dist = torch.pdist(e_batch, p=2)
        return torch.log(torch.exp(-2.0 * pairwise_dist.pow(2)).mean() + EPSILON)

    def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))

        h_rel = self._mask_head(h_emb, relation)

        q = self._compose(h_rel, r_emb)

        # ALIGN(x, y) = ||x - y||_2^2
        return (q - t_emb).norm(p=2, dim=-1).pow(2).mean()

    def total_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        align = self.align_loss(head, relation, tail)
        uni = self.batch_uniformity_loss(head, tail)
        return align + self.gamma * uni

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Inference distance used for scoring in link prediction / triple classification
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))

        h_rel = self._mask_head(h_emb, relation)

        q = self._compose(h_rel, r_emb)
        return (q - t_emb).norm(p=2, dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # If your tester relies on temp scaling for logits
        temp = getattr(self, 'temp', 1.0)
        return -self.forward(head, relation ,tail) / temp

    def constraint(self) -> None:
        # Constraints are handled dynamically via L2 normalization during the forward pass.
        pass


class DirectAUKG(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DirectAU_KG'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate

        self.model = DirectAU_KGModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> tuple[float, int]:
        
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        best_epoch = -1
        patience_counter = 0

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            
            # Shuffle data
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx].to(config.device)
            relation = relation[rand_idx].to(config.device)
            tail = tail[rand_idx].to(config.device)
            
            for start_idx in range(0, n_train, self.n_batch):
                end_idx = min(start_idx + self.n_batch, n_train)
                
                h_batch = head[start_idx:end_idx]
                r_batch = relation[start_idx:end_idx]
                t_batch = tail[start_idx:end_idx]
                
                self.model.zero_grad()
                
                loss = self.model.total_loss(h_batch, r_batch, t_batch)
                
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item() * (end_idx - start_idx)

            avg_loss = epoch_loss / n_train
            logging.info('Epoch %d/%d, Total Loss=%f', epoch + 1, self.n_epoch, avg_loss)

            # Evaluation and Early Stopping
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                logging.info('Running validation at epoch %d...', epoch + 1)
                test_perf = tester()
                logging.info('Validation finished at epoch %d: MRR=%f', epoch + 1, test_perf)
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