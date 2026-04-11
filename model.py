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
        self.encoder_name = model_config.get('encoder_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_length = model_config.get('max_length', 64)
        self.encode_batch_size = model_config.get('encode_batch_size', 64)
        self.temp = model_config.get('temp', 1.0)
        self.uniformity_max_samples = int(model_config.get('uniformity_max_samples', 0))
        self.uniformity_chunk_size = max(1, int(model_config.get('uniformity_chunk_size', 256)))
        self.forward_chunk_size = max(1, int(model_config.get('forward_chunk_size', 65536)))

        self.n_entity, self.n_relation = n_entity, n_relation
        self.entity_texts = entity_texts
        self.relation_texts = relation_texts

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.hr_encoder = AutoModel.from_pretrained(self.encoder_name)
        self.t_encoder = AutoModel.from_pretrained(self.encoder_name)

        self.use_gradient_checkpointing = model_config.get('gradient_checkpointing', True)
        self.freeze_lower_layers = int(model_config.get('freeze_lower_layers', 0))
        self.freeze_embeddings = bool(model_config.get('freeze_embeddings', False))
        self.freeze_tail_encoder = bool(model_config.get('freeze_tail_encoder', False))

        if self.use_gradient_checkpointing:
            self.hr_encoder.gradient_checkpointing_enable()
            if not self.freeze_tail_encoder:
                self.t_encoder.gradient_checkpointing_enable()
            # Required when embeddings are frozen: checkpointed blocks need at least one grad-carrying input.
            if hasattr(self.hr_encoder, 'enable_input_require_grads'):
                self.hr_encoder.enable_input_require_grads()
            if (not self.freeze_tail_encoder) and hasattr(self.t_encoder, 'enable_input_require_grads'):
                self.t_encoder.enable_input_require_grads()
            # Disable KV cache when checkpointing to prevent incompatibilities and extra memory.
            if hasattr(self.hr_encoder.config, 'use_cache'):
                self.hr_encoder.config.use_cache = False
            if hasattr(self.t_encoder.config, 'use_cache'):
                self.t_encoder.config.use_cache = False

        if self.freeze_embeddings:
            if hasattr(self.hr_encoder, 'embeddings'):
                for param in self.hr_encoder.embeddings.parameters():
                    param.requires_grad = False
            if hasattr(self.t_encoder, 'embeddings'):
                for param in self.t_encoder.embeddings.parameters():
                    param.requires_grad = False

        if self.freeze_lower_layers > 0:
            if hasattr(self.hr_encoder, 'encoder') and hasattr(self.hr_encoder.encoder, 'layer'):
                hr_layers = len(self.hr_encoder.encoder.layer)
                n_freeze_hr = min(self.freeze_lower_layers, hr_layers)
                for i in range(n_freeze_hr):
                    for param in self.hr_encoder.encoder.layer[i].parameters():
                        param.requires_grad = False
            if hasattr(self.t_encoder, 'encoder') and hasattr(self.t_encoder.encoder, 'layer'):
                t_layers = len(self.t_encoder.encoder.layer)
                n_freeze_t = min(self.freeze_lower_layers, t_layers)
                for i in range(n_freeze_t):
                    for param in self.t_encoder.encoder.layer[i].parameters():
                        param.requires_grad = False

        if self.freeze_tail_encoder:
            for param in self.t_encoder.parameters():
                param.requires_grad = False

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
        tail_trainable = any(p.requires_grad for p in self.t_encoder.parameters()) and self.training
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
            if tail_trainable:
                hidden = self.t_encoder(**encoded).last_hidden_state
            else:
                with torch.no_grad():
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

        # Optional subsampling keeps peak memory predictable for large batches.
        if self.uniformity_max_samples > 0 and x.size(0) > self.uniformity_max_samples:
            idx = torch.randperm(x.size(0), device=x.device)[:self.uniformity_max_samples]
            x = x[idx]

        # UNI(x) = log(mean(exp(-2 * ||x_i - x_j||_2^2))) over i < j
        # Chunked implementation avoids materializing full O(N^2) pairwise tensors.
        n = x.size(0)
        pair_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        pair_count = 0
        chunk = self.uniformity_chunk_size

        for i_start in range(0, n, chunk):
            i_end = min(i_start + chunk, n)
            xi = x[i_start:i_end]
            for j_start in range(i_start, n, chunk):
                j_end = min(j_start + chunk, n)
                xj = x[j_start:j_end]

                dist_sq = (xi.unsqueeze(1) - xj.unsqueeze(0)).pow(2).sum(dim=-1)
                weights = torch.exp(-2 * dist_sq)

                if i_start == j_start:
                    diag = torch.eye(i_end - i_start, device=x.device, dtype=torch.bool)
                    valid = weights.masked_select(~diag)
                    pair_sum = pair_sum + valid.sum() * 0.5
                    pair_count += valid.numel() // 2
                else:
                    pair_sum = pair_sum + weights.sum()
                    pair_count += weights.numel()

        if pair_count == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return torch.log(pair_sum / pair_count)

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

        n_flat = q_inverse.numel()
        if n_flat <= self.forward_chunk_size:
            t_emb = t_unique_emb[t_inverse]
            dist = (q_emb - t_emb).norm(p=2, dim=-1)
            return dist.reshape(flat_shape)

        # Chunked path avoids materializing huge [n_flat, dim] tensors at once.
        dist_parts = []
        for start in range(0, n_flat, self.forward_chunk_size):
            end = min(start + self.forward_chunk_size, n_flat)
            q_idx = q_inverse[start:end]
            t_idx = t_inverse[start:end]
            q_part = q_unique_emb[q_idx]
            t_part = t_unique_emb[t_idx]
            dist_parts.append((q_part - t_part).norm(p=2, dim=-1))
        return torch.cat(dist_parts, dim=0).reshape(flat_shape)

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
        self.grad_accum_steps = max(1, int(self.model_config.get('grad_accum_steps', 1)))
        self.amp_enabled = bool(self.model_config.get('amp', True)) and config.device.type == 'cuda'
        amp_dtype_name = str(self.model_config.get('amp_dtype', 'fp16')).lower()
        self.amp_dtype = torch.float16 if amp_dtype_name == 'fp16' else torch.bfloat16

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

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found in DirectAUKG. "
                "Please reduce freeze settings (freeze_embeddings/freeze_lower_layers)."
            )
        self.opt = OPTIMIZER_MAP[self.optimizer_name](trainable_params, lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        logging.info(
            'DirectAUKG memory settings: amp=%s (%s), grad_accum_steps=%d, gradient_checkpointing=%s, '
            'freeze_embeddings=%s, freeze_lower_layers=%d, freeze_tail_encoder=%s',
            self.amp_enabled,
            'fp16' if self.amp_dtype == torch.float16 else 'bf16',
            self.grad_accum_steps,
            self.model.use_gradient_checkpointing,
            self.model.freeze_embeddings,
            self.model.freeze_lower_layers,
            self.model.freeze_tail_encoder,
        )

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> tuple[float, int]:
        
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        best_epoch = -1
        patience_counter = 0
        warned_no_grad_step = False

        def _has_any_grad() -> bool:
            for group in self.opt.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        return True
            return False

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            self.model.train()
            warned_no_grad_step = False
            
            # Shuffle data
            rand_idx = torch.randperm(n_train)
            head_device = head[rand_idx].to(config.device)
            relation_device = relation[rand_idx].to(config.device)
            tail_device = tail[rand_idx].to(config.device)

            self.opt.zero_grad(set_to_none=True)
            batch_idx = 0
            total_batches = max(1, self.n_batch)

            # Train without corrupted negatives: DirectAU uses alignment + global uniformity.
            for batch_idx, (h_batch, r_batch, t_batch) in enumerate(batch_by_num(
                self.n_batch,
                head_device,
                relation_device,
                tail_device,
                n_sample=n_train,
            ), start=1):
                with torch.autocast(
                    device_type=config.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    # 1. Encode once and reuse for both alignment and uniformity to save memory.
                    q_full = self.model.encode_query(h_batch, r_batch)
                    t_full = self.model.encode_tail(t_batch)
                    loss_align = (q_full - t_full).norm(p=2, dim=-1).pow(2).mean()

                    # 2. Uniformity on unique samples selected from the already encoded batch.
                    q_pairs = torch.stack([h_batch, r_batch], dim=1)
                    _, q_inverse = torch.unique(q_pairs, dim=0, return_inverse=True)
                    q_order = torch.argsort(q_inverse)
                    q_inv_sorted = q_inverse[q_order]
                    q_mask = torch.ones_like(q_inv_sorted, dtype=torch.bool)
                    q_mask[1:] = q_inv_sorted[1:] != q_inv_sorted[:-1]
                    q_unique = q_full[q_order[q_mask]]

                    _, t_inverse = torch.unique(t_batch, return_inverse=True)
                    t_order = torch.argsort(t_inverse)
                    t_inv_sorted = t_inverse[t_order]
                    t_mask = torch.ones_like(t_inv_sorted, dtype=torch.bool)
                    t_mask[1:] = t_inv_sorted[1:] != t_inv_sorted[:-1]
                    t_unique = t_full[t_order[t_mask]]

                    loss_uni_q = self.model.uniformity_loss(q_unique)
                    loss_uni_t = self.model.uniformity_loss(t_unique)
                    loss_uni = 0.5 * (loss_uni_q + loss_uni_t)

                    # 3. Total DirectAU Loss
                    loss = loss_align + (self.model.gamma * loss_uni)

                loss_for_step = loss / self.grad_accum_steps
                self.scaler.scale(loss_for_step).backward()

                should_step = (batch_idx % self.grad_accum_steps == 0)
                grad_status = 'ACCUM'
                if should_step:
                    if _has_any_grad():
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        grad_status = 'STEP'
                    elif not warned_no_grad_step:
                        logging.warning(
                            'Skipping optimizer step because no gradients were produced. '
                            'Current freeze settings may freeze all parameters used in forward.'
                        )
                        warned_no_grad_step = True
                        grad_status = 'STEP_SKIPPED'
                    else:
                        grad_status = 'STEP_SKIPPED'
                    self.opt.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * h_batch.size(0)

                progress_ratio = batch_idx / total_batches
                bar_width = 24
                filled = int(progress_ratio * bar_width)
                bar = ('#' * filled) + ('-' * (bar_width - filled))
                accum_pos = ((batch_idx - 1) % self.grad_accum_steps) + 1
                logging.info(
                    'Epoch %d/%d [%s] batch %d/%d | samples=%d | loss=%.6f | grad=%s (%d/%d)',
                    epoch + 1,
                    self.n_epoch,
                    bar,
                    batch_idx,
                    total_batches,
                    h_batch.size(0),
                    loss.item(),
                    grad_status,
                    accum_pos,
                    self.grad_accum_steps,
                )

            # Flush the remainder when number of mini-batches is not divisible by grad_accum_steps.
            if batch_idx % self.grad_accum_steps != 0:
                if _has_any_grad():
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    logging.info(
                        'Epoch %d/%d flush step after batch %d/%d | grad=STEP',
                        epoch + 1,
                        self.n_epoch,
                        batch_idx,
                        total_batches,
                    )
                elif not warned_no_grad_step:
                    logging.warning(
                        'Skipping optimizer step because no gradients were produced. '
                        'Current freeze settings may freeze all parameters used in forward.'
                    )
                    warned_no_grad_step = True
                    logging.info(
                        'Epoch %d/%d flush step after batch %d/%d | grad=STEP_SKIPPED',
                        epoch + 1,
                        self.n_epoch,
                        batch_idx,
                        total_batches,
                    )
                self.opt.zero_grad(set_to_none=True)

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