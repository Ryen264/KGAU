from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
import logging
import os

import config
from datasets import batch_by_size
from metrics import ranking_metrics, classification_metrics # Added classification_metrics

EPSILON = 1e-30
FILTER_RANKING_PENALTY = 1e30

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.is_distance_based = None
        self.margin = None
        
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)
    
    def ensure_optimizer(self) -> None:
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.parameters(), weight_decay=0.0)

    def constraint(self) -> None:
        pass

    def parameters(self, recurse = True):
        return super().parameters(recurse)

    def pair_loss(self, head_good: torch.Tensor, relation: torch.Tensor, tail_good: torch.Tensor,
                  head_bad: torch.Tensor, tail_bad: torch.Tensor) -> torch.Tensor:
        if not self.is_distance_based:
            raise NotImplementedError("Pairwise loss is only implemented for distance-based models within margin.")
        
        d_good = self.dist(head_good, relation, tail_good)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(d_good - d_bad + self.margin)
    
    def softmax_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        probs = self.prob(head, relation, tail)
        n = probs.size(0)
        device = probs.device

        row_idx = torch.arange(n, device=device)
        truth_probs = torch.log(probs[row_idx, truth] + EPSILON)
        return -truth_probs
    
class BaseModel(object):
    def __init__(self, n_entity: int, n_relation: int):
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.model_path = None 
        self.weight_decay = 0.0
        self.model = None           # type: BaseModule
        self.opt = None             # type: torch.optim.Optimizer
        self.lr = 0.0

        self.dataset = config._config.dataset
        self.task = config._config.task
        self.task_dir = os.path.join('.', 'models', self.dataset, self.task, 'components')
        os.makedirs(self.task_dir, exist_ok=True)
        self.test_batch_size = config._config.test_batch_size

    def load(self, filepath: str) -> None:
        self.model.load_state_dict(torch.load(filepath, map_location=config.device))

    def save(self, filepath: str=None) -> None:
        if filepath is None:
            filepath = self.model_path
        torch.save(self.model.state_dict(), filepath)

    def ensure_optimizer(self) -> None:
        self.model.ensure_optimizer()

    def constraint(self) -> None:
        self.model.constraint()

    def parameters(self, recurse = True):
        return self.model.parameters(recurse)
    
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.score(head, relation, tail)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.dist(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.prob_logit(head, relation, tail)
    
    # --- HELPER FOR BATCH INFERENCE ---
    def _get_batch_scores(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for h, r, t in batch_by_size(self.test_batch_size, heads, relations, tails):
                batch_scores = self.model.score(h.to(config.device), r.to(config.device), t.to(config.device))
                all_scores.append(batch_scores.cpu())
        return torch.cat(all_scores)

    # --- TRIPLE CLASSIFICATION METHODS ---
    def find_thresholds(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Find the best distance thresholds for triple classification using a validation set.
        Returns a dictionary with relation-specific thresholds and a 'global' fallback threshold.
        """
        scores = self._get_batch_scores(heads, relations, tails)
        labels = labels.cpu()
        relations = relations.cpu()
        
        thresholds = {}
        is_dist = self.model.is_distance_based

        def get_best_thresh(s, l):
            best_acc = 0.0
            best_t = 0.0
            s_sorted = torch.unique(s) # Evaluate at unique score points
            for t in s_sorted:
                preds = (s <= t).long() if is_dist else (s >= t).long()
                acc = (preds == l).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_t = t.item()
            return best_t

        # 1. Find Global Threshold
        thresholds['global'] = get_best_thresh(scores, labels)
        
        # 2. Find Relation-Specific Thresholds
        unique_relations = torch.unique(relations)
        for r in unique_relations:
            r_mask = (relations == r)
            thresholds[r.item()] = get_best_thresh(scores[r_mask], labels[r_mask])
            
        logging.info(f"Thresholds configured for {len(unique_relations)} specific relations. Global fallback: {thresholds['global']:.4f}")
        return thresholds

    def test_classification(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, thresholds: dict) -> dict:
        """
        Evaluate triple classification on a test set using the learned thresholds.
        """
        scores = self._get_batch_scores(heads, relations, tails)
        labels = labels.cpu()
        relations = relations.cpu()
        
        predictions = torch.zeros_like(labels)
        is_dist = self.model.is_distance_based
        
        # Apply relation-specific thresholds (fallback to global if unseen relation)
        for i in range(len(relations)):
            r = relations[i].item()
            thresh = thresholds.get(r, thresholds.get('global', 0.0))
            
            if is_dist:
                predictions[i] = 1 if scores[i] <= thresh else 0
            else:
                predictions[i] = 1 if scores[i] >= thresh else 0

        # Confidence scores for AUC: if distance-based, smaller distance means higher confidence (so negate)
        conf_scores = -scores if is_dist else scores
        
        metrics = classification_metrics(predictions.tolist(), labels.tolist(), conf_scores.tolist())
        
        # Format metrics for cleaner output
        parts = [f"{k.upper()}: {v:.4f}" for k, v in metrics.items()]
        logging.info(f"Classification metrics: {', '.join(parts)}")
        
        return metrics

    def _score_all_entities_chunked(self, head_id: int, relation_id: int, tail_id: int,
                                    predict: str, chunk_size: int) -> torch.Tensor:
        """
        Score all candidate entities for a single triple in chunks to avoid GPU OOM.
        predict='head' ranks all possible heads for (?, r, t).
        predict='tail' ranks all possible tails for (h, r, ?).
        """
        scores = torch.empty(self.n_entity, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, self.n_entity, chunk_size):
                end = min(start + chunk_size, self.n_entity)
                candidates = torch.arange(start, end, dtype=torch.long, device=config.device)
                rel = torch.full((end - start,), relation_id, dtype=torch.long, device=config.device)

                if predict == 'head':
                    head = candidates
                    tail = torch.full((end - start,), tail_id, dtype=torch.long, device=config.device)
                elif predict == 'tail':
                    head = torch.full((end - start,), head_id, dtype=torch.long, device=config.device)
                    tail = candidates
                else:
                    raise ValueError(f"Unknown predict mode: {predict}")

                chunk_scores = self.model.score(head, rel, tail).detach().float().cpu()
                scores[start:end] = chunk_scores
        return scores

    # --- LINK PREDICTION METHODS ---
    def test_link(self, test_data: list, heads: dict, tails: dict, filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        self.model.eval()
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        test_data_no_label = test_data[:3]
        count = 0
        chunk_size = getattr(config._config, 'lp_eval_chunk_size', 1024)
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(self.test_batch_size, *test_data_no_label):
                for head, relation, tail in zip(batch_head, batch_relation, batch_tail):
                    head_id, relation_id, tail_id = head.item(), relation.item(), tail.item()

                    head_scores = self._score_all_entities_chunked(
                        head_id=head_id,
                        relation_id=relation_id,
                        tail_id=tail_id,
                        predict='head',
                        chunk_size=chunk_size,
                    )
                    tail_scores = self._score_all_entities_chunked(
                        head_id=head_id,
                        relation_id=relation_id,
                        tail_id=tail_id,
                        predict='tail',
                        chunk_size=chunk_size,
                    )

                    if filt:
                        key_head = (tail_id, relation_id)
                        if key_head in heads and heads[key_head]._nnz() > 1:
                            tmp = head_scores[head_id].item()
                            head_scores += heads[key_head].to_dense() * FILTER_RANKING_PENALTY
                            head_scores[head_id] = tmp
                            
                        key_tail = (head_id, relation_id)
                        if key_tail in tails and tails[key_tail]._nnz() > 1:
                            tmp = tail_scores[tail_id].item()
                            tail_scores += tails[key_tail].to_dense() * FILTER_RANKING_PENALTY
                            tail_scores[tail_id] = tmp

                    head_metrics = ranking_metrics(head_scores, head_id, k_list=k_list)
                    tail_metrics = ranking_metrics(tail_scores, tail_id, k_list=k_list)

                    head_mr = head_metrics['mr']
                    head_mrr = head_metrics['mrr']
                    head_hits = head_metrics['hits']

                    tail_mr = tail_metrics['mr']
                    tail_mrr = tail_metrics['mrr']
                    tail_hits = tail_metrics['hits']

                    mr_total += (head_mr + tail_mr)
                    mrr_total += (head_mrr + tail_mrr)
                    hits_total = [(hits_total[i] + head_hits[i] + tail_hits[i]) for i in range(len(k_list))]
                    count += 2
                    
        mr_rate = mr_total / count
        mrr_rate = mrr_total / count
        hits_rate = [hit_total / count for hit_total in hits_total]
        
        metrics = {}
        metrics['mr'] = mr_rate
        metrics['mrr'] = mrr_rate
        for i in range(len(k_list)):
            metrics[f'hit@{k_list[i]}'] = hits_rate[i]

        # Format metrics for cleaner output
        parts = []
        label_map = {'mr': 'MR', 'mrr': 'MRR'}
        for k, v in metrics.items():
            label = label_map.get(k, k.replace('hit@', 'Hit@'))
            parts.append(f"{label}: {v:.4f}")
        metrics_str = f"Ranking metrics: {', '.join(parts)}\n"
        logging.info(metrics_str)
        return metrics