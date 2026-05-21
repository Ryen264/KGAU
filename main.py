import argparse
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Must be set before first CUDA context initialization.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import config
from data_loader import graph_size, index_entity_relation, read_data
from datasets import BernCorrupter, convert_data_to_no_label, sparse_heads_tails
from graph_context import LinkGraph
from model import DirectAUKG


@dataclass
class ExperimentResult:
	model_name: str
	best_valid_mrr: float
	best_epoch: int
	link_metrics: Dict[str, float]
	classification_metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and compare DirectAUKG vs TransE on WN18RR for link prediction and triple classification."
	)
	parser.add_argument(
		"config_path",
		nargs="?",
		default=None,
		help="Optional positional path to YAML config file (e.g. python main.py config/config_wn18rr.yaml).",
	)
	parser.add_argument("--config", default="./config/config_wn18rr.yaml", help="Path to YAML config file.")
	parser.add_argument("--dataset", default="wn18rr", choices=["wn18rr"], help="Dataset name.")
	parser.add_argument("--data_root", default="./data", help="Root folder that contains dataset files.")
	parser.add_argument("--log_dir", default="./logs", help="Root folder for log files.")
	parser.add_argument("--no_log_to_file", action="store_true", help="Disable writing logs to file.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--gpu", type=int, default=None, help="GPU id. If not set, auto-select.")
	parser.add_argument("--early_stop_patience", type=int, default=-1, help="Early stopping patience. -1 disables it.")
	parser.add_argument(
		"--quick_eval_samples",
		type=int,
		default=1024,
		help="Number of validation triples used for quick sanity-check eval during training. <=0 means full valid set.",
	)
	parser.add_argument(
		"--cls_n_thresholds",
		type=int,
		default=401,
		help="Number of predefined thresholds for validation-based triple classification tuning. <=1 falls back to unique-score thresholds.",
	)

	parser.add_argument("--dim", type=int, default=200, help="Embedding dimension.")
	parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for evaluation.")

	parser.add_argument("--transe_n_epoch", type=int, default=200, help="Epochs for TransE.")
	parser.add_argument("--transe_n_batch", type=int, default=128, help="Mini-batches per epoch for TransE.")
	parser.add_argument("--transe_lr", type=float, default=1e-3, help="Learning rate for TransE.")
	parser.add_argument("--transe_margin", type=float, default=1.0, help="Margin for TransE ranking loss.")
	parser.add_argument("--transe_p", type=int, default=1, choices=[1, 2], help="Norm type for TransE.")
	parser.add_argument("--transe_temp", type=float, default=1.0, help="Temperature for TransE logits.")

	parser.add_argument("--direct_n_epoch", type=int, default=200, help="Epochs for DirectAUKG.")
	parser.add_argument("--direct_n_batch", type=int, default=128, help="Mini-batches per epoch for DirectAUKG.")
	parser.add_argument("--direct_lr", type=float, default=1e-3, help="Learning rate for DirectAUKG.")
	parser.add_argument("--direct_gamma", type=float, default=1.0, help="Uniformity loss weight for DirectAUKG.")
	parser.add_argument("--direct_encoder_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-BERT encoder name for DirectAUKG bi-encoder towers.")
	parser.add_argument("--direct_max_length", type=int, default=64, help="Max token length for DirectAUKG text encoding.")
	parser.add_argument("--direct_encode_batch_size", type=int, default=64, help="Tokenizer/encoder micro-batch size for DirectAUKG.")
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
		"test_batch_size": max(1, args.test_batch_size),
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
			"n_batch": min(args.direct_n_batch, 8),
			"epoch_per_test": 5,
			"optimizer": "Adam",
			"learning_rate": args.direct_lr,
			"dim": args.dim,
			"gamma": args.direct_gamma,
			"encoder_name": args.direct_encoder_name,
			"max_length": min(args.direct_max_length, 32),
			"encode_batch_size": min(args.direct_encode_batch_size, 4),
			"compose_mode": args.direct_compose,
			"amp": True,
			"amp_dtype": "fp16",
			"gradient_checkpointing": True,
			"grad_accum_steps": 16,
			"freeze_embeddings": True,
			"freeze_lower_layers": 6,
			"uniformity_max_samples": 64,
			"uniformity_chunk_size": 128,
			"forward_chunk_size": 32768,
			"use_link_graph": True,
			"link_graph_max_neighbors": 10,
			"neighbor_min_tokens": 20,
			"neighbor_text_field": "entity",
			"triplet_masking_for_neighbors": True,
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
	return {
		"train": os.path.join(base_dir, "train.txt"),
		"valid_w_label": os.path.join(base_dir, "valid_w_label.txt"),
		"test_w_label": os.path.join(base_dir, "test_w_label.txt"),
	}


def _decode_wordnet_lemma(raw_lemma: str) -> str:
	lemma = raw_lemma.strip()
	if lemma.startswith("__"):
		lemma = lemma[2:]
	lemma = re.sub(r"_[A-Z]{2}_[0-9]+$", "", lemma)
	lemma = lemma.replace("_", " ")
	return lemma.strip()


def build_text_corpora(args: argparse.Namespace, kb_index) -> Tuple[List[str], List[str]]:
	if args.dataset == "wn18rr":
		definitions_path = os.path.join(args.data_root, "wn18rr", "wordnet-mlj12-definitions.txt")
		definition_map: Dict[str, str] = {}
		if os.path.exists(definitions_path):
			with open(definitions_path, encoding="utf-8") as f:
				for line in f:
					parts = line.rstrip("\n").split("\t", 2)
					if len(parts) < 2:
						continue
					synset_id = parts[0].strip()
					lemma = _decode_wordnet_lemma(parts[1])
					definition = parts[2].strip() if len(parts) > 2 else ""
					definition_map[synset_id] = f"{lemma}. {definition}" if definition else lemma

		entity_texts = [definition_map.get(symbol, symbol) for symbol in kb_index.entity_list]
		relation_texts = [rel.strip("_").replace("_", " ") for rel in kb_index.relation_list]
		return entity_texts, relation_texts

	entity_texts = list(kb_index.entity_list)
	relation_texts = [rel.strip("_").replace("_", " ") for rel in kb_index.relation_list]
	return entity_texts, relation_texts


def validate_paths(paths: Dict[str, str]) -> None:
	missing = [p for p in paths.values() if not os.path.exists(p)]
	if missing:
		raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def to_tensor_triplets(data: Tuple[list, list, list]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	h, r, t = data
	return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t)


def to_tensor_triplets_with_labels(
	data: Tuple[list, list, list, list],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	h, r, t, y = data
	return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t), torch.LongTensor(y)


def train_and_evaluate(
	model_name: str,
	model,
	train_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	valid_lp_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	test_lp_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
	valid_cls_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
	test_cls_triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
	n_entity: int,
	early_stop_patience: int,
	quick_eval_samples: int,
	cls_n_thresholds: int,
) -> ExperimentResult:
	def _sample_triplets(
		triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
		n_samples: int,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		head, relation, tail = triplets
		total = head.size(0)
		if n_samples <= 0 or n_samples >= total:
			return triplets
		idx = torch.randperm(total)[:n_samples]
		return head[idx], relation[idx], tail[idx]

	train_lists = tuple(x.tolist() for x in train_triplets)
	valid_quick_triplets = _sample_triplets(valid_lp_triplets, quick_eval_samples)
	valid_quick_lists = tuple(x.tolist() for x in valid_quick_triplets)
	test_lists = tuple(x.tolist() for x in test_lp_triplets)
	valid_full_lists = tuple(x.tolist() for x in valid_lp_triplets)

	eval_heads_valid, eval_tails_valid = sparse_heads_tails(n_entity, train_lists, valid_full_lists, None)
	eval_heads_test, eval_tails_test = sparse_heads_tails(n_entity, train_lists, valid_full_lists, test_lists)

	corrupter = None
	if getattr(model, "uses_negative_sampling", True):
		corrupter = BernCorrupter(train_lists, n_entity, model.n_relation)

	valid_h, valid_r, valid_t, valid_y = valid_cls_triplets
	test_h, test_r, test_t, test_y = test_cls_triplets

	def valid_epoch_tester(epoch_idx: int, total_epochs: int) -> float:
		valid_lp_metrics = model.test_link(valid_lp_triplets, eval_heads_valid, eval_tails_valid, filt=True)
		valid_thresholds = model.find_thresholds(
			valid_h,
			valid_r,
			valid_t,
			valid_y,
			n_thresholds=cls_n_thresholds,
		)
		valid_cls_metrics = model.test_classification(
			valid_h,
			valid_r,
			valid_t,
			valid_y,
			valid_thresholds,
		)
		msg_lp = (
			f"Validation Link Prediction (epoch {epoch_idx}/{total_epochs}): "
			f"MR={valid_lp_metrics['mr']:.4f}, MRR={valid_lp_metrics['mrr']:.4f}, "
			f"Hit@1={valid_lp_metrics['hit@1']:.4f}, Hit@3={valid_lp_metrics['hit@3']:.4f}, "
			f"Hit@10={valid_lp_metrics['hit@10']:.4f}"
		)
		msg_cls = (
			f"Validation Triple Classification (epoch {epoch_idx}/{total_epochs}): "
			f"ACC={valid_cls_metrics['accuracy']:.4f}, PREC={valid_cls_metrics['precision']:.4f}, "
			f"REC={valid_cls_metrics['recall']:.4f}, F1={valid_cls_metrics['f1']:.4f}, "
			f"PR_AUC={valid_cls_metrics['pr_auc']:.4f}, ROC_AUC={valid_cls_metrics['roc_auc']:.4f}"
		)
		print(msg_lp)
		print(msg_cls)
		logging.info(msg_lp)
		logging.info(msg_cls)
		return float(valid_lp_metrics["mrr"])

	def test_epoch_tester(epoch_idx: int, total_epochs: int) -> None:
		test_lp_metrics = model.test_link(test_lp_triplets, eval_heads_test, eval_tails_test, filt=True)
		thresholds = model.find_thresholds(
			valid_h,
			valid_r,
			valid_t,
			valid_y,
			n_thresholds=cls_n_thresholds,
		)
		test_cls_metrics = model.test_classification(
			test_h,
			test_r,
			test_t,
			test_y,
			thresholds,
		)
		msg_lp = (
			f"Test Link Prediction (epoch {epoch_idx}/{total_epochs}): "
			f"MR={test_lp_metrics['mr']:.4f}, MRR={test_lp_metrics['mrr']:.4f}, "
			f"Hit@1={test_lp_metrics['hit@1']:.4f}, Hit@3={test_lp_metrics['hit@3']:.4f}, "
			f"Hit@10={test_lp_metrics['hit@10']:.4f}"
		)
		msg_cls = (
			f"Test Triple Classification (epoch {epoch_idx}/{total_epochs}): "
			f"ACC={test_cls_metrics['accuracy']:.4f}, PREC={test_cls_metrics['precision']:.4f}, "
			f"REC={test_cls_metrics['recall']:.4f}, F1={test_cls_metrics['f1']:.4f}, "
			f"PR_AUC={test_cls_metrics['pr_auc']:.4f}, ROC_AUC={test_cls_metrics['roc_auc']:.4f}"
		)
		print(msg_lp)
		print(msg_cls)
		logging.info(msg_lp)
		logging.info(msg_cls)

	best_valid_mrr, best_epoch = model.train(
		train_triplets,
		corrupter,
		valid_epoch_tester,
		test_tester=test_epoch_tester,
		early_stop_patience=early_stop_patience,
	)

	# Full evaluation is run only once after training is complete.
	link_metrics = model.test_link(test_lp_triplets, eval_heads_test, eval_tails_test, filt=True)

	thresholds = model.find_thresholds(
		valid_h,
		valid_r,
		valid_t,
		valid_y,
		n_thresholds=cls_n_thresholds,
	)
	classification_metrics = model.test_classification(
		test_h,
		test_r,
		test_t,
		test_y,
		thresholds,
	)

	return ExperimentResult(
		model_name=model_name,
		best_valid_mrr=best_valid_mrr,
		best_epoch=best_epoch,
		link_metrics=link_metrics,
		classification_metrics=classification_metrics,
	)


def print_summary(results: Tuple[ExperimentResult, ...]) -> None:
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
			"Triple Classification (test_w_label): "
			f"ACC={res.classification_metrics['accuracy']:.4f}, "
			f"PREC={res.classification_metrics['precision']:.4f}, "
			f"REC={res.classification_metrics['recall']:.4f}, "
			f"F1={res.classification_metrics['f1']:.4f}, "
			f"PR_AUC={res.classification_metrics['pr_auc']:.4f}, "
			f"ROC_AUC={res.classification_metrics['roc_auc']:.4f}"
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


		# Load preprocessed data
	import json
	dataset_dir = os.path.dirname(paths["train"])
	with open(os.path.join(dataset_dir, "train.txt.json"), encoding="utf-8") as f:
		train_examples = json.load(f)
	with open(os.path.join(dataset_dir, "valid_w_label.txt.json"), encoding="utf-8") as f:
		valid_examples = json.load(f)
	with open(os.path.join(dataset_dir, "test_w_label.txt.json"), encoding="utf-8") as f:
		test_examples = json.load(f)

	# Entities and relations
	with open(os.path.join(dataset_dir, "entities.json"), encoding="utf-8") as f:
		entities = json.load(f)
	with open(os.path.join(dataset_dir, "relations.json"), encoding="utf-8") as f:
		relation_map = json.load(f)

	n_entity = len(entities)
	n_relation = len(relation_map)
	logging.info("Graph size: n_entity=%d, n_relation=%d", n_entity, n_relation)

	entity_texts = [e["entity_desc"] if "entity_desc" in e else e["entity"] for e in entities]
	entity_names = [e["entity"] for e in entities]
	relation_texts = list(relation_map.values())

	# Helper to convert list of dicts to tensors
	def examples_to_triplets(examples):
		h = [e["head_id"] for e in examples]
		r = [e["relation"] for e in examples]
		t = [e["tail_id"] for e in examples]
		return h, r, t

	# Build id mapping
	entity_id_map = {e["entity_id"]: idx for idx, e in enumerate(entities)}
	relation_id_map = {v: idx for idx, v in enumerate(relation_map.values())}

	def encode_triplets(examples):
		h, r, t = examples_to_triplets(examples)
		h = [entity_id_map[x] for x in h]
		r = [relation_id_map[x] for x in r]
		t = [entity_id_map[x] for x in t]
		return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t)

	def encode_triplets_with_labels(examples):
		h, r, t = examples_to_triplets(examples)
		y = [e["label"] if "label" in e else 1 for e in examples]
		h = [entity_id_map[x] for x in h]
		r = [relation_id_map[x] for x in r]
		t = [entity_id_map[x] for x in t]
		return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(t), torch.LongTensor(y)

	train_triplets = encode_triplets(train_examples)
	valid_triplets = encode_triplets(valid_examples)
	test_triplets = encode_triplets(test_examples)
	valid_cls_triplets = encode_triplets_with_labels(valid_examples)
	test_cls_triplets = encode_triplets_with_labels(test_examples)

	train_h_list = train_triplets[0].detach().cpu().tolist()
	train_t_list = train_triplets[2].detach().cpu().tolist()
	link_graph = LinkGraph(train_h_list, train_t_list)

	direct_model = DirectAUKG(
		n_entity,
		n_relation,
		entity_texts,
		relation_texts,
		link_graph=link_graph,
		entity_names=entity_names,
	)

	direct_result = train_and_evaluate(
		model_name="DirectAUKG",
		model=direct_model,
		train_triplets=train_triplets,
		valid_lp_triplets=valid_triplets,
		test_lp_triplets=test_triplets,
		valid_cls_triplets=valid_cls_triplets,
		test_cls_triplets=test_cls_triplets,
		n_entity=n_entity,
		early_stop_patience=args.early_stop_patience,
		quick_eval_samples=args.quick_eval_samples,
		cls_n_thresholds=args.cls_n_thresholds,
	)

	print_summary((direct_result,))


if __name__ == "__main__":
	main()
