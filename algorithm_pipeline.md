# DirectAU-DistMult: Algorithm Pipeline

## Algorithm 1: Training Stage

```
Algorithm TRAIN(D_train, n_entity, n_relation, config)
Input:
    D_train: Training triplets (h, r, t)
    n_entity: Number of entities
    n_relation: Number of relations
    config: Configuration (dim, learning_rate, n_epoch, batch_size, gamma, temp)
Output:
    θ: Trained model parameters
    best_perf: Best validation MRR
    best_epoch: Epoch with best validation performance

1: Initialize embeddings
    - E_h ← Embedding(n_entity, dim)
    - E_r ← Embedding(n_relation, dim)
    - W_r ← Embedding(n_relation, dim)  // Attention weights
    - Initialize weights uniformly in [-init_range, init_range]
    
2: Initialize optimizer ← Adam(θ, lr)
3: best_perf ← 0.0
4: best_state ← None
5: patience_count ← 0

6: for epoch = 1 to n_epoch do
7:    total_loss ← 0.0
8:    Shuffle D_train
9:    
10:   for each batch B in D_train do
11:       h_batch, r_batch, t_batch ← B
12:       batch_size ← |B|
13:       
14:       // Normalize embeddings
15:       h ← NORMALIZE(E_h[h_batch])
16:       r ← NORMALIZE(E_r[r_batch])
17:       t ← NORMALIZE(E_t[t_batch])
18:       w ← W_r[r_batch]
19:       
20:       // Composition: query q = normalize((h ⊙ sigmoid(w)) ⊙ r)
21:       q ← NORMALIZE((h ⊙ SIGMOID(w)) ⊙ r)
22:       
23:       // Alignment loss: ||q - t||²
24:       L_align ← ||q - t||²_2 / batch_size
25:       
26:       // Uniformity loss: log(E[exp(-2||e_i - e_j||²)])
27:       unique_entities ← UNIQUE(h_batch ∪ t_batch)
28:       e_emb ← NORMALIZE(E_h[unique_entities])
29:       dist_sq ← PDIST(e_emb, p=2)²
30:       L_unif ← log(MEAN(exp(-2 * dist_sq)))
31:       
32:       // Combined loss
33:       L ← L_align + γ * L_unif
34:       
35:       // Backward pass
36:       ∇θ ← GRADIENT(L)
37:       θ ← UPDATE(θ, ∇θ, optimizer)
38:       
39:       total_loss ← total_loss + L
40:   end for
41:   
42:   avg_loss ← total_loss / |D_train|
43:   LOG("Epoch " + epoch + ", Loss = " + avg_loss)
44:   
45:   // Validation every epoch_per_test epochs
46:   if (epoch mod epoch_per_test == 0) then
47:       valid_mrr ← EVAL_LINK(D_valid, model)
48:       if valid_mrr > best_perf then
49:           best_perf ← valid_mrr
50:           best_epoch ← epoch
51:           best_state ← CLONE(θ)
52:           patience_count ← 0
53:       else
54:           patience_count ← patience_count + 1
55:       end if
56:       
57:       if early_stop_patience > 0 AND patience_count ≥ early_stop_patience then
58:           BREAK
59:       end if
60:   end if
61: end for
62:
63: // Load best model
64: if best_state ≠ None then
65:     θ ← best_state
66: end if
67: SAVE(θ)
68: return θ, best_perf, best_epoch
```

## Algorithm 2: Validation & Thresholding

```
Algorithm FIND_THRESHOLDS(D_valid, model)
Input:
    D_valid: Validation triplets with labels (h, r, t, y) where y ∈ {0, 1}
    model: Trained model
Output:
    thresholds: Dictionary {r → threshold_r, 'global' → threshold_global}

1: scores ← BATCH_SCORES(model, D_valid)  // Get model scores for all validation triplets
2: labels ← EXTRACT_LABELS(D_valid)
3: thresholds ← {}

4: function FIND_BEST_THRESHOLD(scores_subset, labels_subset):
5:     best_acc ← 0.0
6:     best_thresh ← 0.0
7:     unique_scores ← UNIQUE(scores_subset)
8:     
9:     for each thresh in unique_scores do
10:        if is_distance_based then
11:            predictions ← (scores_subset ≤ thresh ? 1 : 0)
12:        else
13:            predictions ← (scores_subset ≥ thresh ? 1 : 0)
14:        end if
15:        
16:        accuracy ← MEAN(predictions == labels_subset)
17:        if accuracy > best_acc then
18:            best_acc ← accuracy
19:            best_thresh ← thresh
20:        end if
21:    end for
22:    return best_thresh
23: end function

24: // Find global threshold
25: thresholds['global'] ← FIND_BEST_THRESHOLD(scores, labels)

26: // Find relation-specific thresholds
27: unique_relations ← UNIQUE(EXTRACT_RELATIONS(D_valid))
28: for each r in unique_relations do
29:     mask_r ← (EXTRACT_RELATIONS(D_valid) == r)
29:     scores_r ← scores[mask_r]
30:     labels_r ← labels[mask_r]
31:     thresholds[r] ← FIND_BEST_THRESHOLD(scores_r, labels_r)
32: end for

33: return thresholds
```

## Algorithm 3: Link Prediction Inference

```
Algorithm EVAL_LINK(D_test, model, heads_filter, tails_filter, filt=True)
Input:
    D_test: Test triplets (h, r, t)
    model: Trained model
    heads_filter: Sparse matrix of (t, r) → {head candidates}
    tails_filter: Sparse matrix of (h, r) → {tail candidates}
    filt: Whether to filter training/validation triplets
Output:
    metrics: {mr, mrr, hit@1, hit@3, hit@10}

1: mr_total ← 0.0
2: mrr_total ← 0.0
3: hits_total ← [0, 0, 0]  // For k ∈ {1, 3, 10}
4: count ← 0
5: k_list ← [1, 3, 10]

6: for each batch B in D_test do
7:     batch_size ← |B|
8:     
9:     for each (h, r, t) in B do
10:        // Head prediction: score all entities as head for (?, r, t)
11:        head_scores ← []
12:        for h' in [0, n_entity) do
13:            head_scores[h'] ← model.SCORE(h', r, t)
14:        end for
15:        
16:        // Filtering: increase scores of training/validation heads if filt=True
17:        if filt then
18:            if (t, r) in heads_filter AND |heads_filter(t, r)| > 1 then
19:                key_val ← head_scores[h]
20:                head_scores ← head_scores + heads_filter(t, r) * PENALTY
21:                head_scores[h] ← key_val  // Restore original score for ground truth
22:            end if
23:        end if
24:        
25:        head_metrics ← RANKING_METRICS(head_scores, h, k_list) // sort scores (ascending)
26:        mr_head ← head_metrics['mr']
27:        mrr_head ← head_metrics['mrr']
28:        hits_head ← head_metrics['hits']
29:        
30:        // Tail prediction: score all entities as tail for (h, r, ?)
31:        tail_scores ← []
32:        for t' in [0, n_entity) do
33:            tail_scores[t'] ← model.SCORE(h, r, t')
34:        end for
35:        
36:        // Filtering for tails
37:        if filt then
38:            if (h, r) in tails_filter AND |tails_filter(h, r)| > 1 then
39:                key_val ← tail_scores[t]
40:                tail_scores ← tail_scores + tails_filter(h, r) * PENALTY
41:                tail_scores[t] ← key_val
42:            end if
43:        end if
44:        
45:        tail_metrics ← RANKING_METRICS(tail_scores, t, k_list)
46:        mr_tail ← tail_metrics['mr']
47:        mrr_tail ← tail_metrics['mrr']
48:        hits_tail ← tail_metrics['hits']
49:        
50:        // Aggregate metrics
51:        mr_total ← mr_total + mr_head + mr_tail
52:        mrr_total ← mrr_total + mrr_head + mrr_tail
53:        hits_total ← hits_total + hits_head + hits_tail
54:        count ← count + 2
55:    end for
56: end for

57: // Average metrics
58: mr_avg ← mr_total / count
59: mrr_avg ← mrr_total / count
60: hits_avg ← hits_total / count

61: // Compute Hits@k for each k
62: metrics ← {'mr': mr_avg, 'mrr': mrr_avg}
63: for i = 0 to len(k_list)-1 do
64:     metrics['hit@' + k_list[i]] ← hits_avg[i]
64: end for

65: return metrics

// Helper function: Compute ranking metrics for a single triple
function RANKING_METRICS(scores, target, k_list):
    sorted_idx ← ARGSORT(scores)
    target_rank ← POSITION(sorted_idx, target) + 1
    hits_at_k ← [(1 if target_rank ≤ k else 0) for k in k_list]
    return {'mr': target_rank, 'mrr': 1/target_rank, 'hits': hits_at_k}
end function
```

## Algorithm 4: Triple Classification Inference

```
Algorithm EVAL_CLASSIFY(D_test, model, thresholds)
Input:
    D_test: Test triplets with labels (h, r, t, y)
    model: Trained model
    thresholds: Dictionary of relation-specific thresholds
Output:
    metrics: {accuracy, precision, recall, f1, pr_auc, roc_auc}

1: scores ← BATCH_SCORES(model, D_test)  // Get model scores
2: labels ← EXTRACT_LABELS(D_test)
3: relations ← EXTRACT_RELATIONS(D_test)
4: predictions ← []
5: conf_scores ← []

6: for i = 1 to |D_test| do
7:     r ← relations[i]
8:     // Use relation-specific threshold if available, else use global
9:     thresh ← GET(thresholds, r, thresholds['global'])
10:    
11:    if is_distance_based then
12:        pred ← (scores[i] ≤ thresh ? 1 : 0)
13:        conf ← -scores[i]  // Negate: smaller distance = higher confidence
14:    else
15:        pred ← (scores[i] ≥ thresh ? 1 : 0)
16:        conf ← scores[i]  // Higher score = higher confidence
17:    end if
18:    
19:    predictions[i] ← pred
20:    conf_scores[i] ← conf
21: end for

22: // Compute classification metrics
23: tp ← SUM((predictions == 1) ∧ (labels == 1))
24: tn ← SUM((predictions == 0) ∧ (labels == 0))
25: fp ← SUM((predictions == 1) ∧ (labels == 0))
26: fn ← SUM((predictions == 0) ∧ (labels == 1))

27: accuracy ← (tp + tn) / (tp + tn + fp + fn)
28: precision ← tp / (tp + fp)
29: recall ← tp / (tp + fn)
30: f1 ← 2 * (precision * recall) / (precision + recall)

31: // AUC computation (requires sklearn)
32: roc_auc ← ROC_AUC_SCORE(labels, conf_scores)
33: pr_auc ← PR_AUC_SCORE(labels, conf_scores)

34: metrics ← {
35:     'accuracy': accuracy,
36:     'precision': precision,
37:     'recall': recall,
38:     'f1': f1,
39:     'roc_auc': roc_auc,
40:     'pr_auc': pr_auc
41: }

42: return metrics
```

## Legend & Notation

- `E_h`, `E_r`, `W_r`: Entity, relation, and attention weight embeddings
- `NORMALIZE(x)`: L2 normalization: $x / (||x||_2 + \epsilon)$
- `⊙`: Element-wise (Hadamard) product
- `SIGMOID`: Sigmoid activation function
- `PDIST`: Pairwise distance matrix
- `||·||_2`: L2 norm
- `θ`: Model parameters (embeddings and weights)
- `is_distance_based`: Flag indicating if model uses distance-based scoring
- `PENALTY`: Large value (1e30) to filter known triplets
- `MR`: Mean Rank
- `MRR`: Mean Reciprocal Rank
- `Hits@k`: Percentage of targets ranked in top k
