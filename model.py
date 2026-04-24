import torch
import torch.nn as nn
import logging
import os
from typing import Tuple
from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop

import config
from base_model import BaseModule, BaseModel

OPTIMIZER_MAP = {
    'Adam': Adam,
    'SGD': SGD,
    'Adagrad': Adagrad,
    'Adadelta': Adadelta,
    'RMSprop': RMSprop,
}

EPSILON = 1e-8

class DirectAU_TransDModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'DirectAU-TransD'

        self.entity_dim = config.dim
        self.relation_dim = config.get('relation_dim', self.entity_dim)
        self.gamma = config.get('gamma', 1.0)
        self.temp = config.get('temp', 1.0)

        self.n_entity, self.n_relation = n_entity, n_relation

        # Meaning vectors.
        self.entity_embed = nn.Embedding(self.n_entity, self.entity_dim)
        self.relation_embed = nn.Embedding(self.n_relation, self.relation_dim)

        # Projection vectors.
        self.entity_proj_embed = nn.Embedding(self.n_entity, self.entity_dim)
        self.relation_proj_embed = nn.Embedding(self.n_relation, self.relation_dim)

        # Relation attention mask weights w_r in relation space.
        self.relation_attn_embed = nn.Embedding(self.n_relation, self.relation_dim)

        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        # Initialize all meaning/projection vectors with Uniform(-6/sqrt(k), 6/sqrt(k)).
        ent_range = 6.0 / (self.entity_dim ** 0.5)
        rel_range = 6.0 / (self.relation_dim ** 0.5)
        self.entity_embed.weight.data.uniform_(-ent_range, ent_range)
        self.relation_embed.weight.data.uniform_(-rel_range, rel_range)
        self.entity_proj_embed.weight.data.uniform_(-ent_range, ent_range)
        self.relation_proj_embed.weight.data.uniform_(-rel_range, rel_range)
        self.relation_attn_embed.weight.data.uniform_(-rel_range, rel_range)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Strict L2 normalization used by DirectAU-TransD."""
        return x / (x.norm(p=2, dim=-1, keepdim=True) + EPSILON)

    def _identity_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute I^{m x n} x where x is in R^n.
        If m < n, truncate. If m > n, zero-pad.
        """
        if self.relation_dim == self.entity_dim:
            return x
        if self.relation_dim < self.entity_dim:
            return x[..., :self.relation_dim]
        pad_size = self.relation_dim - self.entity_dim
        pad = torch.zeros(*x.shape[:-1], pad_size, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def _project_entities(self, e: torch.Tensor, e_p: torch.Tensor, r_p: torch.Tensor) -> torch.Tensor:
        """
        TransD projection:
        e_perp = (r_p e_p^T + I^{m x n}) e = I^{m x n} e + r_p * <e_p, e>
        """
        identity_part = self._identity_map(e)
        scalar = torch.sum(e_p * e, dim=-1, keepdim=True)
        return identity_part + r_p * scalar

    def _aligned_components(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)

        h_p = self.entity_proj_embed(head)
        t_p = self.entity_proj_embed(tail)
        r_p = self.relation_proj_embed(relation)
        w_r = self.relation_attn_embed(relation)

        h_perp = self._project_entities(h, h_p, r_p)
        t_perp = self._project_entities(t, t_p, r_p)

        h_bar = self._normalize(h_perp)
        r_bar = self._normalize(r)
        t_bar = self._normalize(t_perp)

        # Attention mask on projected head in relation space, then strict re-normalization.
        h_mask = self._normalize(h_bar * torch.sigmoid(w_r))

        q = self._normalize(h_mask + r_bar)
        return q, t_bar

    def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Local alignment term: ||q_bar - t_bar||_2^2 for each triple in the batch.
        q, t_bar = self._aligned_components(head, relation, tail)
        return (q - t_bar).norm(p=2, dim=-1).pow(2)

    def uniformity_loss(self, unique_entities: torch.Tensor) -> torch.Tensor:
        """Batch uniformity with log Gaussian potential over i != j pairs."""
        if unique_entities.numel() < 2:
            return torch.zeros((), device=unique_entities.device)

        # Uniformity is computed from raw meaning vectors, then normalized.
        e_emb = self._normalize(self.entity_embed(unique_entities))
        dist_sq = torch.pdist(e_emb, p=2).pow(2)
        return dist_sq.mul(-2).exp().mean().log()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Inference distance follows the same aligned geometry used in training.
        q, t_bar = self._aligned_components(head, relation, tail)
        return (q - t_bar).norm(p=2, dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail) / self.temp

    def constraint(self) -> None:
        # Constraints are enforced via strict normalization in forward/align computation.
        pass


class DirectAUKG(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DirectAU-KG'

        # Support both legacy and canonical config section names.
        if self.model_type in config._config:
            self.model_config = config._config[self.model_type]
        elif 'DirectAU_KG' in config._config:
            self.model_config = config._config['DirectAU_KG']
        elif 'DirectAU-TransD' in config._config:
            self.model_config = config._config['DirectAU-TransD']
        else:
            raise KeyError("Config must contain one of: 'DirectAU-KG', 'DirectAU_KG', or 'DirectAU-TransD'.")

        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.batch_size = self.model_config.get('batch_size', 128)
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate
        self.weight_decay = self.model_config.get('weight_decay', 0.0)

        self.model = DirectAU_TransDModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> tuple[float, int]:
        """Train Updated TransD with DirectAU alignment + uniformity objective."""
        
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        best_epoch = -1
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            
            # Shuffle data
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx].to(config.device)
            relation = relation[rand_idx].to(config.device)
            tail = tail[rand_idx].to(config.device)
            
            # Minibatch training per DirectAU algorithm
            for start_idx in range(0, n_train, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_train)
                batch_size = end_idx - start_idx
                
                h_batch = head[start_idx:end_idx]
                r_batch = relation[start_idx:end_idx]
                t_batch = tail[start_idx:end_idx]
                
                self.model.zero_grad()
                
                # Step 1: Calculate alignment loss for all triples in batch
                loss_align_per_triple = self.model.align_loss(h_batch, r_batch, t_batch)
                loss_align_total = loss_align_per_triple.sum()
                
                # Step 2: Calculate uniformity loss over unique entities in batch
                unique_entities = torch.cat((h_batch, t_batch)).unique()
                loss_unif = self.model.uniformity_loss(unique_entities)
                
                # Step 3: Normalize alignment loss by batch size and compute total loss
                loss_align_normalized = loss_align_total / batch_size
                loss = loss_align_normalized + (self.model.gamma * loss_unif)
                
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item() * batch_size

            avg_loss = epoch_loss / n_train
            logging.info('Epoch %d/%d, Total Loss=%f', epoch + 1, self.n_epoch, avg_loss)

            # Evaluation and Early Stopping
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()
                if (test_perf > best_perf):
                    best_perf = test_perf
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, early_stop_patience)
                break
                
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.save(self.model_path)
        return best_perf, best_epoch