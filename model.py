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
        self.gamma = getattr(config, 'gamma', 1.0) # Backward-compatible default weight.
        self.gamma_h = getattr(config, 'gamma_h', self.gamma)
        self.gamma_t = getattr(config, 'gamma_t', self.gamma)
        self.compose_mode = getattr(config, 'compose_mode', 'mul') # 'mul' (Hadamard) or 'add'

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            # Standard initialization, but no renorm_ here since we enforce 
            # strict projection to the unit hypersphere in the forward pass.
            param.data.normal_(0, 1 / param.size(1) ** 0.5)

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

    def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))

        q = self._compose(h_emb, r_emb)
        
        # ALIGN(x, y) = ||x - y||_2^2
        return (q - t_emb).norm(p=2, dim=-1).pow(2).mean()

    def uniformity_loss(self, unique_entities: torch.Tensor) -> torch.Tensor:
        if unique_entities.numel() < 2:
            return torch.zeros((), device=unique_entities.device)

        e_emb = self._normalize(self.entity_embed(unique_entities))
        
        # UNI(x) = log(mean(exp(-2 * ||x_i - x_j||_2^2)))
        # torch.pdist computes flattened pairwise distances without the zero-diagonals
        dist_sq = torch.pdist(e_emb, p=2).pow(2)
        return dist_sq.mul(-2).exp().mean().log()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Inference distance used for scoring in link prediction / triple classification
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))
        
        q = self._compose(h_emb, r_emb)
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
            
            # Custom batching logic: We drop the corrupter and negative samples entirely
            for start_idx in range(0, n_train, self.n_batch):
                end_idx = min(start_idx + self.n_batch, n_train)
                
                h_batch = head[start_idx:end_idx]
                r_batch = relation[start_idx:end_idx]
                t_batch = tail[start_idx:end_idx]
                
                self.model.zero_grad()
                
                # 1. Calculate Alignment Loss
                loss_align = self.model.align_loss(h_batch, r_batch, t_batch)
                
                # 2. Calculate Uniformity Loss separately for head and tail entities
                unique_heads = h_batch.unique()
                unique_tails = t_batch.unique()
                loss_uni_h = self.model.uniformity_loss(unique_heads)
                loss_uni_t = self.model.uniformity_loss(unique_tails)
                loss_uni = (self.model.gamma_h * loss_uni_h) + (self.model.gamma_t * loss_uni_t)
                
                # 3. Total DirectAU Loss
                loss = loss_align + (self.model.gamma * loss_uni)
                
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item() * (end_idx - start_idx)

            avg_loss = epoch_loss / n_train
            logging.info('Epoch %d/%d, Total Loss=%f', epoch + 1, self.n_epoch, avg_loss)

            # Evaluation and Early Stopping
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