import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop

import config
from base_model import BaseModule, BaseModel
from datasets import batch_by_size

OPTIMIZER_MAP = {
	'Adam': Adam,
	'SGD': SGD,
	'Adagrad': Adagrad,
	'Adadelta': Adadelta,
	'RMSprop': RMSprop,
}


class TransDModule(BaseModule):
	def __init__(self, n_entity: int, n_relation: int, config: config.config):
		super().__init__()
		self.model_type = 'TransD'

		self.entity_dim = config.dim
		self.relation_dim = config.get('relation_dim', self.entity_dim)
		self.margin = config.margin
		self.p = config.p
		self.temp = config.temp

		self.n_entity, self.n_relation = n_entity, n_relation

		# Meaning vectors.
		self.entity_embed = nn.Embedding(self.n_entity, self.entity_dim)
		self.relation_embed = nn.Embedding(self.n_relation, self.relation_dim)

		# Projection vectors.
		self.entity_proj_embed = nn.Embedding(self.n_entity, self.entity_dim)
		self.relation_proj_embed = nn.Embedding(self.n_relation, self.relation_dim)

		self.is_distance_based = True
		self.init_weight()

	def init_weight(self) -> None:
		ent_range = 6.0 / (self.entity_dim ** 0.5)
		rel_range = 6.0 / (self.relation_dim ** 0.5)

		self.entity_embed.weight.data.uniform_(-ent_range, ent_range)
		self.relation_embed.weight.data.uniform_(-rel_range, rel_range)
		self.entity_proj_embed.weight.data.uniform_(-ent_range, ent_range)
		self.relation_proj_embed.weight.data.uniform_(-rel_range, rel_range)

		# Meaning vectors follow TransE-style norm constraints.
		self.entity_embed.weight.data.renorm_(2, 0, 1)
		self.relation_embed.weight.data.renorm_(2, 0, 1)

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
		TransD mapping:
		e_perp = (r_p e_p^T + I^{m x n}) e
			   = I^{m x n} e + r_p * <e_p, e>
		"""
		identity_part = self._identity_map(e)
		scalar = torch.sum(e_p * e, dim=-1, keepdim=True)
		projected = identity_part + r_p * scalar

		# Keep projected vectors inside the unit ball.
		norm = torch.norm(projected, p=2, dim=-1, keepdim=True)
		return projected / torch.clamp(norm, min=1.0)

	def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		h = self.entity_embed(head)
		t = self.entity_embed(tail)
		r = self.relation_embed(relation)

		h_p = self.entity_proj_embed(head)
		t_p = self.entity_proj_embed(tail)
		r_p = self.relation_proj_embed(relation)

		h_perp = self._project_entities(h, h_p, r_p)
		t_perp = self._project_entities(t, t_p, r_p)

		return torch.norm(h_perp + r - t_perp, p=self.p, dim=-1)

	def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return self.forward(head, relation, tail)

	def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return self.forward(head, relation, tail)

	def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
		return -self.forward(head, relation, tail) / self.temp

	def constraint(self) -> None:
		# Algorithm constraints for meaning vectors.
		self.entity_embed.weight.data.renorm_(2, 0, 1)
		self.relation_embed.weight.data.renorm_(2, 0, 1)


class TransD(BaseModel):
	def __init__(self, n_entity: int, n_relation: int):
		super().__init__(n_entity, n_relation)
		self.model_type = 'TransD'
		self.model_config = config._config[self.model_type]
		self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

		self.n_epoch = self.model_config.n_epoch
		self.batch_size = self.model_config.get('batch_size', 128)
		self.epoch_per_test = self.model_config.epoch_per_test

		self.optimizer_name = self.model_config.optimizer
		self.lr = self.model_config.learning_rate

		self.model = TransDModule(self.n_entity, self.n_relation, self.model_config)
		self.model.to(config.device)
		self.is_distance_based = self.model.is_distance_based
		self.margin = self.model.margin
		self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr)

	def train(
		self,
		train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
		corrupter,
		tester,
		early_stop_patience: int = -1,
	) -> tuple[float, int]:
		head, relation, tail = train_data
		n_train = len(head)
		best_perf = 0.0
		best_epoch = -1
		best_state_dict = None
		patience_counter = 0

		for epoch in range(self.n_epoch):
			epoch_loss = 0
			rand_idx = torch.randperm(n_train)
			head = head[rand_idx]
			relation = relation[rand_idx]
			tail = tail[rand_idx]

			self.model.constraint()
			head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)

			head_device = head.to(config.device)
			relation_device = relation.to(config.device)
			tail_device = tail.to(config.device)
			head_corrupted = head_corrupted.to(config.device)
			tail_corrupted = tail_corrupted.to(config.device)

			for h0, r, t0, h1, t1 in batch_by_size(
				self.batch_size,
				head_device,
				relation_device,
				tail_device,
				head_corrupted,
				tail_corrupted,
				n_sample=n_train,
			):
				self.model.zero_grad()
				loss = torch.sum(self.model.pair_loss(h0, r, t0, h1, t1))
				loss.backward()
				self.opt.step()
				epoch_loss += loss.item()

			logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)

			if (self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0):
				test_perf = tester()
				if test_perf > best_perf:
					best_perf = test_perf
					best_epoch = epoch + 1
					best_state_dict = {
						k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
					}
					patience_counter = 0
				else:
					patience_counter += 1

				if early_stop_patience > 0 and patience_counter >= early_stop_patience:
					logging.info(
						'Early stopping triggered at epoch %d (patience=%d)',
						epoch + 1,
						early_stop_patience,
					)
					break

		if best_state_dict is not None:
			self.model.load_state_dict(best_state_dict)
		self.save(self.model_path)
		return best_perf, best_epoch
