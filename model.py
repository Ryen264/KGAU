import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
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


class DirectAU_DistMultModule(BaseModule):
	def __init__(self, n_entity: int, n_relation: int, config: config.config):
		super().__init__()
		self.model_type = 'DistMult'

		self.dim = config.dim
		self.gamma = config.get('gamma', 1.0)
		self.temp = config.get('temp', 1.0)

		self.n_entity, self.n_relation = n_entity, n_relation
		self.entity_embed = nn.Embedding(self.n_entity, self.dim)
		self.relation_embed = nn.Embedding(self.n_relation, self.dim)
		self.relation_attn = nn.Embedding(self.n_relation, self.dim)
		self.is_distance_based = True
		self.init_weight()

	def init_weight(self) -> None:
		init_range = 6.0 / (self.dim ** 0.5)
		self.entity_embed.weight.data.uniform_(-init_range, init_range)
		self.relation_embed.weight.data.uniform_(-init_range, init_range)
		self.relation_attn.weight.data.uniform_(-init_range, init_range)

	def _normalize(self, x: torch.Tensor) -> torch.Tensor:
		return x / (x.norm(p=2, dim=-1, keepdim=True) + EPSILON)

	def _compose(self, h: torch.Tensor, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
		h_mask = self._normalize(h * torch.sigmoid(w))
		q_raw = h_mask * r
		return self._normalize(q_raw)

	def _aligned_components(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self._normalize(self.entity_embed(head))
		r = self._normalize(self.relation_embed(relation))
		t = self._normalize(self.entity_embed(tail))
		w_r = self.relation_attn(relation)

		q = self._compose(h, r, w_r)
		return q, t

	def align_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		q, t = self._aligned_components(head, relation, tail)
		return (q - t).norm(p=2, dim=-1).pow(2)

	def uniformity_loss(self, unique_entities: torch.Tensor) -> torch.Tensor:
		if unique_entities.numel() < 2:
			return torch.zeros((), device=unique_entities.device)

		e_emb = self._normalize(self.entity_embed(unique_entities))
		dist_sq = torch.pdist(e_emb, p=2).pow(2)
		return dist_sq.mul(-2).exp().mean().log()

	def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		q, t = self._aligned_components(head, relation, tail)
		return (q - t).norm(p=2, dim=-1).pow(2)

	def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return self.forward(head, relation, tail)

	def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return self.forward(head, relation, tail)

	def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return -self.forward(head, relation, tail) / self.temp

	def constraint(self) -> None:
		pass


class DirectAU_DistMult(BaseModel):
	def __init__(self, n_entity: int, n_relation: int):
		super().__init__(n_entity, n_relation)
		self.model_type = 'DistMult'

		if 'DistMult' in config._config:
			self.model_config = config._config['DistMult']
		elif 'DirectAU-DistMult' in config._config:
			self.model_config = config._config['DirectAU-DistMult']
		elif 'DirectAU_KG' in config._config:
			self.model_config = config._config['DirectAU_KG']
		elif 'DirectAU-KG' in config._config:
			self.model_config = config._config['DirectAU-KG']
		else:
			raise KeyError("Config must contain one of: 'DistMult', 'DirectAU-DistMult', 'DirectAU_KG', or 'DirectAU-KG'.")

		self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

		self.n_epoch = self.model_config.n_epoch
		self.batch_size = self.model_config.get('batch_size', 128)
		self.epoch_per_test = self.model_config.epoch_per_test

		self.optimizer_name = self.model_config.optimizer
		self.lr = self.model_config.learning_rate
		self.weight_decay = self.model_config.get('weight_decay', 0.0)

		self.model = DirectAU_DistMultModule(self.n_entity, self.n_relation, self.model_config)
		self.model.to(config.device)
		self.is_distance_based = self.model.is_distance_based
		self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

	def train(
		self,
		train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
		corrupter,
		tester,
		early_stop_patience: int = -1,
	) -> tuple[float, int]:
		"""Train DirectAU-DistMult with alignment and uniformity objectives."""
		head, relation, tail = train_data
		n_train = len(head)
		best_perf = 0.0
		best_epoch = -1
		best_state_dict = None
		patience_counter = 0

		for epoch in range(self.n_epoch):
			epoch_loss = 0.0
			rand_idx = torch.randperm(n_train)
			head = head[rand_idx].to(config.device)
			relation = relation[rand_idx].to(config.device)
			tail = tail[rand_idx].to(config.device)

			for start_idx in range(0, n_train, self.batch_size):
				end_idx = min(start_idx + self.batch_size, n_train)
				batch_size = end_idx - start_idx

				h_batch = head[start_idx:end_idx]
				r_batch = relation[start_idx:end_idx]
				t_batch = tail[start_idx:end_idx]

				self.model.zero_grad()

				loss_align_total = self.model.align_loss(h_batch, r_batch, t_batch).sum()
				unique_entities = torch.cat((h_batch, t_batch)).unique()
				loss_unif = self.model.uniformity_loss(unique_entities)

				loss_align_normalized = loss_align_total / batch_size
				loss = loss_align_normalized + (self.model.gamma * loss_unif)

				loss.backward()
				self.opt.step()
				epoch_loss += loss.item() * batch_size

			avg_loss = epoch_loss / n_train
			logging.info('Epoch %d/%d, Total Loss=%f', epoch + 1, self.n_epoch, avg_loss)

			if (self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0):
				test_perf = tester()
				if test_perf > best_perf:
					best_perf = test_perf
					best_epoch = epoch + 1
					best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
					patience_counter = 0
				else:
					patience_counter += 1

			if early_stop_patience > 0 and patience_counter >= early_stop_patience:
				logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, early_stop_patience)
				break

		if best_state_dict is not None:
			self.model.load_state_dict(best_state_dict)
		self.save(self.model_path)
		return best_perf, best_epoch


# Backwards-compatible alias for existing callers.
DirectAUKG = DirectAU_DistMult