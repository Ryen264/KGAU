import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

import config
from data_loader import graph_size, index_entity_relation, read_data
from datasets import BernCorrupter, sparse_heads_tails
from model import DirectAUKG
from transe import TransE


@dataclass
class ExperimentResult:
	model_name: str
	best_valid_mrr: float
	best_epoch: int
	link_metrics: Dict[str, float]
	cls_metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and compare DirectAUKG vs TransE on WN18RR for link prediction and triple classification."
	)
	parser.add_argument(
		"config_path",
		nargs="?",
		default=None,
		help="Optional positional path to YAML config file (e.g., python main.py config/config_wn18rr.yaml).",
	)
	parser.add_argument("--config", default="./config/config_wn18rr.yaml", help="Path to YAML config file.")
	parser.add_argument("--dataset", default="wn18rr", choices=["wn18rr"], help="Dataset name.")
	parser.add_argument("--data_root", default="./data", help="Root folder that contains dataset files.")
	parser.add_argument("--log_dir", default="./logs", help="Root folder for log files.")
	parser.add_argument("--no_log_to_file", action="store_true", help="Disable writing logs to file.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--gpu", type=int, default=None, help="GPU id. If not set, auto-select.")
	parser.add_argument("--early_stop_patience", type=int, default=-1, help="Early stopping patience. -1 disables it.")

	parser.add_argument("--dim", type=int, default=200, help="Embedding dimension.")
	parser.add_argument("--test_batch_size", type=int, default=256, help="Batch size for evaluation.")

	parser.add_argument("--transe_n_epoch", type=int, default=200, help="Epochs for TransE.")
	parser.add_argument("--transe_n_batch", type=int, default=128, help="Mini-batches per epoch for TransE.")
	parser.add_argument("--transe_lr", type=float, default=1e-3, help="Learning rate for TransE.")
	parser.add_argument("--transe_margin", type=float, default=1.0, help="Margin for TransE ranking loss.")
	parser.add_argument("--transe_p", type=int, default=1, choices=[1, 2], help="Norm type for TransE.")
	parser.add_argument("--transe_temp", type=float, default=1.0, help="Temperature for TransE logits.")

	parser.add_argument("--direct_n_epoch", type=int, default=200, help="Epochs for DirectAUKG.")
	parser.add_argument("--direct_n_batch", type=int, default=128, help="Mini-batches per epoch for DirectAUKG.")
	parser.add_argument("--direct_lr", type=float, default=1e-3, help="Learning rate for DirectAUKG.")
	parser.add_argument("--direct_gamma", type=float, default=0.1, help="Uniformity weight gamma for DirectAUKG.")
	parser.add_argument("--direct_compose", default="mul", choices=["mul", "add"], help="Composition mode for DirectAUKG.")

	args = parser.parse_args()
	if args.config_path:
		args.config = args.config_path
	return args


def setup_logging(args: argparse.Namespace) -> str:
	root_logger = logging.getLogger()
	root_logger.handlers.clear()
	root_logger.setLevel(logging.INFO)
	formatter = logging.Formatter("%(module)15s %(asctime)s %(message)s", datefmt="%H:%M:%S")

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	root_logger.addHandler(console_handler)

	log_file_path = ""
	if not args.no_log_to_file:
		log_task_dir = os.path.join(args.log_dir, args.dataset, "comparison")
		os.makedirs(log_task_dir, exist_ok=True)
		ts = time.strftime("%y%m%d-%H%M%S")
		log_file_path = os.path.join(log_task_dir, f"compare_directaukg_transe_{ts}.log")
		file_handler = logging.FileHandler(log_file_path)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)

	return log_file_path


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _to_cfg(obj):
	if isinstance(obj, dict):
		return config.ConfigDict({k: _to_cfg(v) for k, v in obj.items()})
	if isinstance(obj, list):
		return [_to_cfg(v) for v in obj]
	return obj


def build_runtime_config(args: argparse.Namespace) -> None:
	runtime_cfg = {
		"dataset": args.dataset,
		"task": "comparison",
		"test_batch_size": args.test_batch_size,
		"lp_eval_chunk_size": 1024,
		"log": {
			"to_file": False,
			"dump_config": False,
			"prefix": "kgau",
		},
		"TransE": {
			"model_file": "transe.pt",
			"n_epoch": args.transe_n_epoch,
			"n_batch": args.transe_n_batch,
			"epoch_per_test": 5,
			"optimizer": "Adam",
			"learning_rate": args.transe_lr,
			"dim": args.dim,
			"margin": args.transe_margin,
			"p": args.transe_p,
			"temp": args.transe_temp,
		},
		"DirectAU_KG": {
			"model_file": "directaukg.pt",
			"n_epoch": args.direct_n_epoch,
			"n_batch": args.direct_n_batch,
			"epoch_per_test": 5,
			"optimizer": "Adam",
			"learning_rate": args.direct_lr,
			"dim": args.dim,
			"gamma": args.direct_gamma,
			"compose_mode": args.direct_compose,
		},
	}
	config._config = _to_cfg(runtime_cfg)


def load_config(args: argparse.Namespace) -> None:
	if os.path.exists(args.config):
		cfg = config.config(args.config)

		# Backward-compatibility: model code expects DirectAU_KG.
		if "DirectAU_KG" not in cfg:
			if "DirectAUKG" in cfg:
				cfg["DirectAU_KG"] = cfg["DirectAUKG"]
			else:
				raise KeyError("Config must contain 'DirectAU_KG' or 'DirectAUKG'.")

		if "dataset" in cfg:
			args.dataset = cfg["dataset"]
		if "test_batch_size" not in cfg:
			cfg["test_batch_size"] = args.test_batch_size
	else:
		logging.warning("Config file not found at %s. Falling back to runtime defaults.", args.config)
		build_runtime_config(args)


def build_paths(args: argparse.Namespace) -> Dict[str, str]:
	base_dir = os.path.join(args.data_root, args.dataset)
	labeled_dir = os.path.join(args.data_root, f"{args.dataset}_w_labels")
	return {
		"train": os.path.join(base_dir, "train.txt"),
		"valid": os.path.join(base_dir, "valid.txt"),
		"test": os.path.join(base_dir, "test.txt"),
		"valid_cls": os.path.join(labeled_dir, "valid.txt"),
		"test_cls": os.path.join(labeled_dir, "test.txt"),
	}


def validate_paths(paths: Dict[str, str]) -> None:
	missing = [p for p in paths.values() if not os.path.exists(p)]
	if missing:
		raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def to_tensor_triplets(data: Tuple[list, list, list]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	h, r, t = data
	return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t)


def to_tensor_quadruples(data: Tuple[list, list, list, list]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	h, r, t, y = data
	return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t), torch.LongTensor(y)


def train_and_evaluate(
	model_name: str,
	model,
	train_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	valid_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	test_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	valid_cls: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
	test_cls: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
	n_entity: int,
	early_stop_patience: int,
) -> ExperimentResult:
	train_lists = tuple(x.tolist() for x in train_triplets)
	valid_lists = tuple(x.tolist() for x in valid_triplets)
	test_lists = tuple(x.tolist() for x in test_triplets)

	eval_heads_valid, eval_tails_valid = sparse_heads_tails(n_entity, train_lists, valid_lists, None)
	eval_heads_test, eval_tails_test = sparse_heads_tails(n_entity, train_lists, valid_lists, test_lists)

	corrupter = BernCorrupter(train_lists, n_entity, model.n_relation)

	def valid_link_tester() -> float:
		valid_metrics = model.test_link(valid_triplets, eval_heads_valid, eval_tails_valid, filt=True)
		return float(valid_metrics["mrr"])

	best_valid_mrr, best_epoch = model.train(
		train_triplets,
		corrupter,
		valid_link_tester,
		early_stop_patience=early_stop_patience,
	)

	link_metrics = model.test_link(test_triplets, eval_heads_test, eval_tails_test, filt=True)

	val_h, val_r, val_t, val_y = valid_cls
	thresholds = model.find_thresholds(val_h, val_r, val_t, val_y)

	test_h, test_r, test_t, test_y = test_cls
	cls_metrics = model.test_classification(test_h, test_r, test_t, test_y, thresholds)

	return ExperimentResult(
		model_name=model_name,
		best_valid_mrr=best_valid_mrr,
		best_epoch=best_epoch,
		link_metrics=link_metrics,
		cls_metrics=cls_metrics,
	)


def print_summary(results: Tuple[ExperimentResult, ExperimentResult]) -> None:
	lines = ["", "=== Comparison On WN18RR ==="]
	for res in results:
		lines.append("")
		lines.append(f"[{res.model_name}]")
		lines.append(f"Best valid MRR: {res.best_valid_mrr:.4f} (epoch={res.best_epoch})")
		lines.append(
			"Link Prediction (test): "
			f"MR={res.link_metrics['mr']:.4f}, "
			f"MRR={res.link_metrics['mrr']:.4f}, "
			f"Hit@1={res.link_metrics['hit@1']:.4f}, "
			f"Hit@3={res.link_metrics['hit@3']:.4f}, "
			f"Hit@10={res.link_metrics['hit@10']:.4f}"
		)
		lines.append(
			"Triple Classification (test): "
			f"Acc={res.cls_metrics['accuracy']:.4f}, "
			f"Prec={res.cls_metrics['precision']:.4f}, "
			f"Rec={res.cls_metrics['recall']:.4f}, "
			f"F1={res.cls_metrics['f1']:.4f}, "
			f"PR-AUC={res.cls_metrics['pr_auc']:.4f}, "
			f"ROC-AUC={res.cls_metrics['roc_auc']:.4f}"
		)

	for line in lines:
		print(line)
		if line:
			logging.info(line)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	load_config(args)
	log_file_path = setup_logging(args)
	if log_file_path:
		logging.info("Writing logs to %s", log_file_path)

	gpu_id = args.gpu if args.gpu is not None else config.select_gpu()
	config.device = config.set_device(gpu_id)

	paths = build_paths(args)
	validate_paths(paths)

	kb_index = index_entity_relation(
		paths["train"],
		paths["valid"],
		paths["test"],
		paths["valid_cls"],
		paths["test_cls"],
	)
	n_entity, n_relation = graph_size(kb_index)
	logging.info("Graph size: n_entity=%d, n_relation=%d", n_entity, n_relation)

	train_triplets = to_tensor_triplets(read_data(paths["train"], kb_index))
	valid_triplets = to_tensor_triplets(read_data(paths["valid"], kb_index))
	test_triplets = to_tensor_triplets(read_data(paths["test"], kb_index))
	valid_cls = to_tensor_quadruples(read_data(paths["valid_cls"], kb_index, with_label=True))
	test_cls = to_tensor_quadruples(read_data(paths["test_cls"], kb_index, with_label=True))

	direct_model = DirectAUKG(n_entity, n_relation)
	transe_model = TransE(n_entity, n_relation)

	direct_result = train_and_evaluate(
		model_name="DirectAUKG",
		model=direct_model,
		train_triplets=train_triplets,
		valid_triplets=valid_triplets,
		test_triplets=test_triplets,
		valid_cls=valid_cls,
		test_cls=test_cls,
		n_entity=n_entity,
		early_stop_patience=args.early_stop_patience,
	)

	transe_result = train_and_evaluate(
		model_name="TransE",
		model=transe_model,
		train_triplets=train_triplets,
		valid_triplets=valid_triplets,
		test_triplets=test_triplets,
		valid_cls=valid_cls,
		test_cls=test_cls,
		n_entity=n_entity,
		early_stop_patience=args.early_stop_patience,
	)

	print_summary((direct_result, transe_result))


if __name__ == "__main__":
	main()
