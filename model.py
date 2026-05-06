import torch
import torch.nn as nn
import logging
import os
from typing import Tuple
from torch.optim import Adam, SGD, Adagrad, RMSprop

import config as config
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
        self.model_type = 'DirectAU-KG'

        self.dim = config.dim
        # configuration values are read at the higher-level model wrapper

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.relation_attn = nn.Embedding(self.n_relation, self.dim)
        self.relation_attn_bias = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        # Inference uses dot-product on normalized vectors (higher is better)
        self.is_distance_based = False
        self.init_weight()

    def init_weight(self) -> None:
        # Initialize embeddings with Uniform(-6/√k, 6/√k) per DirectAU algorithm
        init_range = 6.0 / (self.dim ** 0.5)
        self.relation_embed.weight.data.uniform_(-init_range, init_range)
        self.relation_attn.weight.data.fill_(2.0)
        self.relation_attn_bias.weight.data.zero_()
        self.entity_embed.weight.data.uniform_(-init_range, init_range)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Projects vectors onto the unit hypersphere."""
        return x / (x.norm(p=2, dim=-1, keepdim=True) + EPSILON)

    def _compose(self, h: torch.Tensor, r: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Applies relation-conditioned attention mask on head, then translates and normalizes."""
        # Relation-conditioned gate: m_r = sigmoid(w_r ⊙ r + b_r)
        m = torch.sigmoid(w * r + b)
        h_masked = h * m
        q_raw = h_masked + r
        return self._normalize(q_raw)

    def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Calculate alignment loss per DirectAU algorithm: sum of ||q - t||_2^2 for each triple."""
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))
        w_rel = self.relation_attn(relation)
        b_rel = self.relation_attn_bias(relation)

        q = self._compose(h_emb, r_emb, w_rel, b_rel)
        
        # Alignment loss = ||q - t||_2^2 per triple
        return (q - t_emb).norm(p=2, dim=-1).pow(2)

    def uniformity_loss(self, unique_entities: torch.Tensor) -> torch.Tensor:
        """Calculate batch uniformity loss per DirectAU algorithm using log Gaussian potential."""
        if unique_entities.numel() < 2:
            return torch.zeros((), device=unique_entities.device)

        e_emb = self._normalize(self.entity_embed(unique_entities))
        
        # Uniformity loss = log(mean(exp(-2 * ||e_i - e_j||_2^2))) over all pairs i != j
        dist_sq = torch.pdist(e_emb, p=2).pow(2)
        return (dist_sq.mul(-2).exp().mean() + EPSILON).log()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Inference distance used for scoring in link prediction / triple classification
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))
        w_rel = self.relation_attn(relation)
        b_rel = self.relation_attn_bias(relation)
        
        q = self._compose(h_emb, r_emb, w_rel, b_rel)
        # Return squared L2 distance (useful for training alignment loss)
        return (q - t_emb).norm(p=2, dim=-1).pow(2)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Dot-product scoring between normalized composed query and candidate embeddings.
        # Supports broadcasted / chunked inference where `head`, `relation`, `tail`
        # can be shaped like [batch, n_candidates].
        h_emb = self._normalize(self.entity_embed(head))
        r_emb = self._normalize(self.relation_embed(relation))
        t_emb = self._normalize(self.entity_embed(tail))
        w_rel = self.relation_attn(relation)
        b_rel = self.relation_attn_bias(relation)

        q = self._compose(h_emb, r_emb, w_rel, b_rel)
        return (q * t_emb).sum(dim=-1)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # If your tester relies on temp scaling for logits
        temp = getattr(self, 'temp', 1.0)
        # Use dot-product logits (higher = more likely positive)
        return self.score(head, relation, tail) / temp

    def constraint(self) -> None:
        # Constraints are handled dynamically via L2 normalization during the forward pass.
        pass


class DirectAUKG(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DirectAU-KG'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.batch_size = self.model_config.get('batch_size', 128)
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate
        self.weight_decay = self.model_config.get('weight_decay', 0.0)

        # Algorithm hyperparameters: support backward-compatible keys
        self.gamma_uni = self.model_config.get('gamma_uni', self.model_config.get('gamma', 1.0))
        self.gamma_neg = self.model_config.get('gamma_neg', 0.0)
        self.epsilon = self.model_config.get('epsilon', EPSILON)

        self.model = DirectAU_KGModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> tuple[float, int]:
        """Train using DirectAU-TransE algorithm: no negative sampling, alignment + uniformity loss."""
        
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        best_epoch = -1
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_unif = 0.0
            epoch_neg = 0.0
            
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

                # Optional negative loss (if corrupter provided and gamma_neg > 0)
                loss_neg = torch.tensor(0.0, device=config.device)
                if (self.gamma_neg > 0.0) and (corrupter is not None):
                    # Generate corrupted candidates on CPU (datasets often return CPU tensors)
                    h_corrupt, t_corrupt = corrupter.corrupt(h_batch.cpu(), r_batch.cpu(), t_batch.cpu())
                    h_corrupt = h_corrupt.to(config.device)
                    t_corrupt = t_corrupt.to(config.device)
                    # q computed from positive head & relation
                    h_emb = self.model._normalize(self.model.entity_embed(h_batch))
                    r_emb = self.model._normalize(self.model.relation_embed(r_batch))
                    w_rel = self.model.relation_attn(r_batch)
                    b_rel = self.model.relation_attn_bias(r_batch)
                    q = self.model._compose(h_emb, r_emb, w_rel, b_rel)

                    # Handle single negative per sample (1D) or multiple (2D)
                    if t_corrupt.dim() == 1:
                        t_neg_emb = self.model._normalize(self.model.entity_embed(t_corrupt))
                        d_neg = (q - t_neg_emb).norm(p=2, dim=-1).pow(2)
                        loss_neg = (torch.exp(-2.0 * d_neg).mean() + self.epsilon).log()
                    else:
                        # shape: [batch, n_neg]
                        t_neg_emb = self.model._normalize(self.model.entity_embed(t_corrupt))
                        # expand q to [batch, 1, dim]
                        q_exp = q.unsqueeze(1)
                        d_neg = (q_exp - t_neg_emb).norm(p=2, dim=-1).pow(2)
                        per_sample = (torch.exp(-2.0 * d_neg).mean(dim=1) + self.epsilon).log()
                        loss_neg = per_sample.mean()

                loss = loss_align_normalized + (self.gamma_uni * loss_unif) + (self.gamma_neg * loss_neg)
                
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item() * batch_size
                epoch_align += loss_align_normalized.item() * batch_size
                epoch_unif += loss_unif.item() * batch_size
                epoch_neg += loss_neg.item() * batch_size

            avg_loss = epoch_loss / n_train
            avg_align = epoch_align / n_train
            avg_unif = epoch_unif / n_train
            avg_neg = epoch_neg / n_train
            logging.info(
                'Epoch %d/%d, Total Loss=%f, Align=%f, Uni=%f, Neg=%f',
                epoch + 1,
                self.n_epoch,
                avg_loss,
                avg_align,
                avg_unif,
                avg_neg,
            )

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