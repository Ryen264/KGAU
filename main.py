import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

import config as config
from data_loader import graph_size, index_entity_relation, read_data
from datasets import BernCorrupter, sparse_heads_tails
from model import DirectAUKG
from transe import TransE

# Global list of gamma pairs for test cases: (gamma_uni, gamma_neg)
GAMMA_PAIRS = [
	(0.0, 0.0),
	(0.0, 1.0),
	(1.0, 0.0),
	(1.0, 1.0),
]


@dataclass
class ExperimentResult:
	model_name: str
	best_valid_mrr: float
	best_epoch: int
	link_metrics: Dict[str, float]
	cls_metrics: Dict[str, float]
	train_time: float
	total_time: float

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and compare TransE vs DirectAU-TransE."
	)
	parser.add_argument(
		"config_path",
		nargs="?",
		default=None,
		help="Optional positional path to YAML config file.",
	)
	parser.add_argument("--config", default="./config/config_wn18rr.yaml", help="Path to YAML config file.")
	parser.add_argument("--dataset", default="wn18rr", choices=["wn18rr", "fb15k237", "wn18"], help="Dataset name.")
	parser.add_argument("--data_root", default="./data", help="Root folder that contains dataset files.")
	parser.add_argument("--log_dir", default="./logs", help="Root folder for log files.")
	parser.add_argument("--no_log_to_file", action="store_true", help="Disable writing logs to file.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--gpu", type=int, default=None, help="GPU id. If not set, auto-select.")
	parser.add_argument("--early_stop_patience", type=int, default=-1, help="Early stopping patience. -1 disables it.")
	parser.add_argument("--dim", type=int, default=200, help="Embedding dimension.")
	parser.add_argument("--test_batch_size", type=int, default=256, help="Batch size for evaluation.")
	parser.add_argument("--n_epoch", type=int, default=200, help="Training epochs.")
	parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
	parser.add_argument("--gamma_uni", type=float, default=None, help="Uniformity weight for DirectAU-TransE (overrides config when set).")
	parser.add_argument("--gamma_neg", type=float, default=None, help="Negative sampling weight for DirectAU-TransE (overrides config when set).")
	parser.add_argument("--models", nargs="+", default=["TransE", "DirectAU-KG"], choices=["TransE", "DirectAU-KG"], help="Models to train and compare.")

	args = parser.parse_args()
	if args.config_path:
		args.config = args.config_path
	return args

def setup_logging(args: argparse.Namespace, run_tag: str) -> str:
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
		log_file_path = os.path.join(log_task_dir, f"{run_tag}_DirectAU-TransE.log")
		file_handler = logging.FileHandler(log_file_path, mode="w")
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

def _clone_cfg(cfg):
	if isinstance(cfg, dict):
		return config.ConfigDict({k: _clone_cfg(v) for k, v in cfg.items()})
	if isinstance(cfg, list):
		return [_clone_cfg(v) for v in cfg]
	return cfg

def build_runtime_config(args: argparse.Namespace) -> None:
	effective_gamma_uni = args.gamma_uni if args.gamma_uni is not None else 1.0
	effective_gamma_neg = args.gamma_neg if args.gamma_neg is not None else 0.0
	runtime_cfg = {
		"dataset": args.dataset,
		"task": "comparison",
		"test_batch_size": args.test_batch_size,
		"log": {
			"to_file": False,
			"dump_config": False,
			"prefix": "kgau",
		},
		"TransE": {
			"model_file": "TransE.mdl",
			"n_epoch": args.n_epoch,
			"batch_size": args.batch_size,
			"epoch_per_test": 5,
			"optimizer": "Adam",
			"learning_rate": args.lr,
			"margin": 1.0,
			"p": 2,
			"dim": args.dim,
			"temp": 1.0,
		},
		"DirectAU-KG": {
			"model_file": "DirectAU-KG.mdl",
			"n_epoch": args.n_epoch,
			"batch_size": args.batch_size,
			"epoch_per_test": 5,
			"optimizer": "Adam",
			"learning_rate": args.lr,
			"dim": args.dim,
			"gamma_uni": effective_gamma_uni,
			"gamma_neg": effective_gamma_neg,
		},
	}
	config._config = _to_cfg(runtime_cfg)

def apply_run_artifact_names(run_tag: str) -> None:
	if "TransE" in config._config:
		config._config["TransE"]["model_file"] = f"{run_tag}_TransE.mdl"
	if "DirectAU-KG" in config._config:
		config._config["DirectAU-KG"]["model_file"] = f"{run_tag}_DirectAU-TransE.mdl"

def load_config(args: argparse.Namespace) -> None:
	if os.path.exists(args.config):
		cfg = config.config(args.config)

		# Backward-compatibility: normalize legacy n_batch to batch_size.
		for _, section_cfg in cfg.items():
			if isinstance(section_cfg, dict) and "batch_size" not in section_cfg and "n_batch" in section_cfg:
				section_cfg["batch_size"] = section_cfg["n_batch"]

		# Backward-compatibility: model code expects DirectAU-KG.
		if "DirectAU-KG" not in cfg:
			if "DirectAUKG" in cfg:
				cfg["DirectAU-KG"] = cfg["DirectAUKG"]
			else:
				raise KeyError("Config must contain 'DirectAU-KG' or 'DirectAUKG'.")

		if "dataset" in cfg:
			args.dataset = cfg["dataset"]
		if "test_batch_size" not in cfg:
			cfg["test_batch_size"] = args.test_batch_size

		# If CLI did not override gamma_uni/gamma_neg, inherit from config for user-facing labels/summary.
		if args.gamma_uni is None:
			direct_cfg = cfg.get("DirectAU-KG", {})
			args.gamma_uni = direct_cfg.get("gamma_uni", direct_cfg.get("gamma", 1.0))
		if args.gamma_neg is None:
			direct_cfg = cfg.get("DirectAU-KG", {})
			args.gamma_neg = direct_cfg.get("gamma_neg", 0.0)
	else:
		logging.warning("Config file not found at %s. Falling back to runtime defaults.", args.config)
		build_runtime_config(args)
		if args.gamma_uni is None:
			args.gamma_uni = 1.0
		if args.gamma_neg is None:
			args.gamma_neg = 0.0

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

	# Time the training
	total_start = time.time()
	train_start = time.time()
	best_valid_mrr, best_epoch = model.train(
		train_triplets,
		corrupter,
		valid_link_tester,
		early_stop_patience=early_stop_patience,
	)
	train_time = time.time() - train_start

	link_metrics = model.test_link(test_triplets, eval_heads_test, eval_tails_test, filt=True)

	val_h, val_r, val_t, val_y = valid_cls
	thresholds = model.find_thresholds(val_h, val_r, val_t, val_y)

	test_h, test_r, test_t, test_y = test_cls
	cls_metrics = model.test_classification(test_h, test_r, test_t, test_y, thresholds)
	
	total_time = time.time() - total_start

	return ExperimentResult(
		model_name=model_name,
		best_valid_mrr=best_valid_mrr,
		best_epoch=best_epoch,
		link_metrics=link_metrics,
		cls_metrics=cls_metrics,
		train_time=train_time,
		total_time=total_time,
	)

def print_summary(results: Tuple[ExperimentResult, ...]) -> None:
	lines = ["", "=" * 80]
	lines.append("TransE vs DirectAU-TransE Performance Comparison")
	lines.append("=" * 80)
	
	for res in results:
		lines.append("")
		lines.append(f"[{res.model_name}]")
		lines.append(f"Best valid MRR: {res.best_valid_mrr:.4f} (epoch={res.best_epoch})")
		lines.append(f"Training time: {res.train_time:.2f}s | Total time: {res.total_time:.2f}s")
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
	
	lines.append("")
	lines.append("=" * 80)
	for line in lines:
		print(line)
		if line:
			logging.info(line)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	
	# If GAMMA_PAIRS has multiple entries, run all gamma pairs
	if len(GAMMA_PAIRS) > 1:
		args.n_epoch = 1000
		args.dim = 100
		all_results = []
		
		for gamma_uni, gamma_neg in GAMMA_PAIRS:
			args.gamma_uni = gamma_uni
			args.gamma_neg = gamma_neg
			
			run_tag = time.strftime("%Y%m%d_%H%M%S")
			# Rebuild config with new gamma values
			config._config = None
			build_runtime_config(args)
			apply_run_artifact_names(run_tag)
			log_file_path = setup_logging(args, run_tag)
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

			set_seed(args.seed)
			logging.info(f"\n{'='*80}")
			logging.info(f"[Test Case] gamma_uni={gamma_uni}, gamma_neg={gamma_neg}")
			logging.info(f"{'='*80}")
			
			model = DirectAUKG(n_entity, n_relation)
			result = train_and_evaluate(
				model_name=f"DirectAU-TransE (gamma_uni={gamma_uni}, gamma_neg={gamma_neg})",
				model=model,
				train_triplets=train_triplets,
				valid_triplets=valid_triplets,
				test_triplets=test_triplets,
				valid_cls=valid_cls,
				test_cls=test_cls,
				n_entity=n_entity,
				early_stop_patience=args.early_stop_patience,
			)
			all_results.append(result)
		
		print_summary(tuple(all_results))
	else:
		run_tag = time.strftime("%Y%m%d_%H%M%S")
		load_config(args)
		apply_run_artifact_names(run_tag)
		log_file_path = setup_logging(args, run_tag)
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

		results = []
		
		# Train and evaluate each selected model
		for model_type in args.models:
			set_seed(args.seed)
			logging.info(f"\n{'='*80}")
			logging.info(f"Training {model_type} model...")
			logging.info(f"{'='*80}")
			
			if model_type == "TransE":
				model = TransE(n_entity, n_relation)
				result = train_and_evaluate(
					model_name="TransE (Canonical)",
					model=model,
					train_triplets=train_triplets,
					valid_triplets=valid_triplets,
					test_triplets=test_triplets,
					valid_cls=valid_cls,
					test_cls=test_cls,
					n_entity=n_entity,
					early_stop_patience=args.early_stop_patience,
				)
			elif model_type == "DirectAU-KG":
				model = DirectAUKG(n_entity, n_relation)
				result = train_and_evaluate(
					model_name=f"DirectAU-TransE (gamma_uni={args.gamma_uni}, gamma_neg={args.gamma_neg})",
					model=model,
					train_triplets=train_triplets,
					valid_triplets=valid_triplets,
					test_triplets=test_triplets,
					valid_cls=valid_cls,
					test_cls=test_cls,
					n_entity=n_entity,
					early_stop_patience=args.early_stop_patience,
				)
			
			results.append(result)

		print_summary(tuple(results))


if __name__ == "__main__":
	main()
