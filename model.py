import torch
import torch.nn as nn
import logging
import os
from typing import List, Tuple
from torch.optim import Adam, SGD, Adagrad, RMSprop
from transformers import AutoModel, AutoTokenizer

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

class DirectAU_KGModule(BaseModule):
    def __init__(
        self,
        n_entity: int,
        n_relation: int,
        model_config: config.config,
        entity_texts: List[str],
        relation_texts: List[str],
    ):
        super().__init__()
        self.model_type = 'DirectAU_KG'

        self.gamma = model_config.get('gamma', 1.0)  # Weight of the uniformity loss
        self.encoder_name = model_config.get('encoder_name', 'bert-base-uncased')
        self.max_length = model_config.get('max_length', 64)
        self.encode_batch_size = model_config.get('encode_batch_size', 64)
        self.temp = model_config.get('temp', 1.0)

        self.n_entity, self.n_relation = n_entity, n_relation
        self.entity_texts = entity_texts
        self.relation_texts = relation_texts

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.hr_encoder = AutoModel.from_pretrained(self.encoder_name)
        self.t_encoder = AutoModel.from_pretrained(self.encoder_name)

        self.dim = self.hr_encoder.config.hidden_size
        self.is_distance_based = True

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Projects vectors onto the unit hypersphere."""
        return x / (x.norm(p=2, dim=-1, keepdim=True) + EPSILON)

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        sum_emb = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return sum_emb / denom

    def _ids_to_texts(self, ids: torch.Tensor, text_table: List[str]) -> List[str]:
        return [text_table[i] for i in ids.detach().cpu().tolist()]

    def _encode_text_pairs(self, left_texts: List[str], right_texts: List[str]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(left_texts), self.encode_batch_size):
            end = start + self.encode_batch_size
            # Let the backbone tokenizer inject separator/special tokens in its native format.
            encoded = self.tokenizer(
                left_texts[start:end],
                right_texts[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            encoded = {k: v.to(config.device) for k, v in encoded.items()}
            hidden = self.hr_encoder(**encoded).last_hidden_state
            pooled = self._mean_pool(hidden, encoded['attention_mask'])
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def _encode_single_texts(self, texts: List[str]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), self.encode_batch_size):
            end = start + self.encode_batch_size
            encoded = self.tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            encoded = {k: v.to(config.device) for k, v in encoded.items()}
            hidden = self.t_encoder(**encoded).last_hidden_state
            pooled = self._mean_pool(hidden, encoded['attention_mask'])
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def encode_query(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        head_texts = self._ids_to_texts(head, self.entity_texts)
        relation_texts = self._ids_to_texts(relation, self.relation_texts)
        q_raw = self._encode_text_pairs(head_texts, relation_texts)
        return self._normalize(q_raw)

    def encode_tail(self, tail: torch.Tensor) -> torch.Tensor:
        tail_texts = self._ids_to_texts(tail, self.entity_texts)
        t_raw = self._encode_single_texts(tail_texts)
        return self._normalize(t_raw)

    def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        q = self.encode_query(head, relation)
        t_emb = self.encode_tail(tail)
        
        # ALIGN(x, y) = ||x - y||_2^2
        return (q - t_emb).norm(p=2, dim=-1).pow(2).mean()

    def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Uniformity is undefined for fewer than 2 vectors; keep training stable.
        if x.size(0) < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        
        # UNI(x) = log(mean(exp(-2 * ||x_i - x_j||_2^2)))
        # torch.pdist computes flattened pairwise distances without the zero-diagonals
        dist_sq = torch.pdist(x, p=2).pow(2)
        return dist_sq.mul(-2).exp().mean().log()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        # Inference distance used for scoring in link prediction / triple classification
        flat_shape = head.shape

        head_flat = head.reshape(-1)
        relation_flat = relation.reshape(-1)
        tail_flat = tail.reshape(-1)

        unique_queries, q_inverse = torch.unique(
            torch.stack([head_flat, relation_flat], dim=1),
            dim=0,
            return_inverse=True,
        )
        q_unique_emb = self.encode_query(unique_queries[:, 0], unique_queries[:, 1])
        q_emb = q_unique_emb[q_inverse]

        unique_tails, t_inverse = torch.unique(tail_flat, return_inverse=True)
        t_unique_emb = self.encode_tail(unique_tails)
        t_emb = t_unique_emb[t_inverse]

        dist = (q_emb - t_emb).norm(p=2, dim=-1)
        return dist.reshape(flat_shape)

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
    def __init__(self, n_entity: int, n_relation: int, entity_texts: List[str], relation_texts: List[str]):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DirectAU_KG'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate

        self.model = DirectAU_KGModule(
            self.n_entity,
            self.n_relation,
            self.model_config,
            entity_texts,
            relation_texts,
        )
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.uses_negative_sampling = False
        
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
            head_device = head[rand_idx].to(config.device)
            relation_device = relation[rand_idx].to(config.device)
            tail_device = tail[rand_idx].to(config.device)
            
            # Train without corrupted negatives: DirectAU uses alignment + global uniformity.
            for h_batch, r_batch, t_batch in batch_by_num(
                self.n_batch,
                head_device,
                relation_device,
                tail_device,
                n_sample=n_train,
            ):
                
                self.model.zero_grad()
                
                # 1. Calculate Alignment Loss
                loss_align = self.model.align_loss(h_batch, r_batch, t_batch)
                
                # 2. Calculate Uniformity Loss on unique queries (h, r) and unique tails t
                unique_queries = torch.unique(torch.stack([h_batch, r_batch], dim=1), dim=0)
                q_batch = self.model.encode_query(unique_queries[:, 0], unique_queries[:, 1])

                unique_tails = torch.unique(t_batch)
                t_batch_unique = self.model.encode_tail(unique_tails)

                loss_uni_q = self.model.uniformity_loss(q_batch)
                loss_uni_t = self.model.uniformity_loss(t_batch_unique)
                loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
                
                # 3. Total DirectAU Loss
                loss = loss_align + (self.model.gamma * loss_uni)
                
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item() * h_batch.size(0)

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