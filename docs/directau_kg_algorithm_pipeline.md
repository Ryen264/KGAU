# DirectAU-KG: Algorithm Pipeline

## Algorithm 1: Training Stage

```
Algorithm TRAIN(D_train, n_entity, n_relation, config)
Input:
    D_train: Training triplets (h, r, t)
    n_entity: Number of entities
    n_relation: Number of relations
    config: Configuration (dim, learning_rate, n_epoch, n_batch, epoch_per_test, gamma, compose_mode, early_stop_patience)
Output:
    θ: Trained model parameters
    best_perf: Best validation MRR
    best_epoch: Epoch with best validation performance

1: Initialize entity embeddings E_e ∈ R^(n_entity × dim)
2: Initialize relation embeddings E_r ∈ R^(n_relation × dim)
3: Initialize relation mask embeddings M_r ∈ R^(n_relation × dim)
4: Set is_distance_based ← True
5: Initialize parameters from N(0, 1/sqrt(dim))
6: Initialize optimizer ← Adam(θ, lr = learning_rate)
7: best_perf ← 0.0
8: best_epoch ← -1
9: patience_count ← 0

10: for epoch = 1 to n_epoch do
11:     Shuffle D_train
12:     epoch_loss ← 0.0
13:
14:     for each mini-batch B = (h_batch, r_batch, t_batch) in D_train do
15:         zero gradients
16:
17:         h ← L2_NORMALIZE(E_e[h_batch])
18:         r ← L2_NORMALIZE(E_r[r_batch])
19:         t ← L2_NORMALIZE(E_e[t_batch])
20:         mask ← SIGMOID(M_r[r_batch])
21:         h_masked ← L2_NORMALIZE(h ⊙ mask)
22:
23:         if compose_mode = 'mul' then
24:             q_raw ← h_masked ⊙ r
25:         else
26:             q_raw ← h_masked + r
27:         end if
28:         q ← L2_NORMALIZE(q_raw)
29:
30:         L_align ← MEAN(||q - t||_2^2)
31:
32:         unique_entities ← UNIQUE(h_batch ∪ t_batch)
33:         if |unique_entities| < 2 then
34:             L_uni ← 0
35:         else
36:             e_uni ← L2_NORMALIZE(E_e[unique_entities])
37:             pairwise_dist ← PDIST(e_uni, p=2)
38:             L_uni ← log(MEAN(exp(-2 * pairwise_dist^2)) + ε)
39:         end if
40:
41:         L ← L_align + gamma * L_uni
42:         backpropagate L
43:         update θ using optimizer
44:
45:         epoch_loss ← epoch_loss + L × |B|
46:     end for
47:
48:     avg_loss ← epoch_loss / |D_train|
49:     LOG(epoch, avg_loss)
50:
51:     if epoch mod epoch_per_test = 0 then
52:         valid_mrr ← EVAL_LINK(D_valid, model)
53:         if valid_mrr > best_perf then
54:             best_perf ← valid_mrr
55:             best_epoch ← epoch
56:             best_state ← COPY(θ)
57:             patience_count ← 0
58:         else
59:             patience_count ← patience_count + 1
60:         end if
61:
62:         if early_stop_patience > 0 and patience_count ≥ early_stop_patience then
63:             BREAK
64:         end if
65:     end if
66: end for

67: if best_state exists then
68:     θ ← best_state
69: end if
70: SAVE(θ)
71: return θ, best_perf, best_epoch
```

## Algorithm 2: Validation & Thresholding

```
Algorithm FIND_THRESHOLDS(D_valid, model)
Input:
    D_valid: Validation quadruples (h, r, t, y) where y ∈ {0, 1}
    model: Trained DirectAU-KG model
Output:
    thresholds: Dictionary {global, r → threshold_r}

1: scores ← BATCH_SCORE(model, D_valid)
2: labels ← EXTRACT_LABELS(D_valid)
3: relations ← EXTRACT_RELATIONS(D_valid)
4: is_dist ← model.is_distance_based
5: thresholds ← empty dictionary

6: function BEST_THRESHOLD(scores_subset, labels_subset):
7:     best_acc ← 0.0
8:     best_thresh ← 0.0
9:     candidate_thresholds ← UNIQUE(scores_subset)
10:
11:    for each thresh in candidate_thresholds do
12:        if is_dist then
13:            preds ← (scores_subset ≤ thresh)
14:        else
15:            preds ← (scores_subset ≥ thresh)
16:        end if
17:
18:        acc ← MEAN(preds = labels_subset)
19:        if acc > best_acc then
20:            best_acc ← acc
21:            best_thresh ← thresh
22:        end if
23:    end for
24:    return best_thresh
25: end function

26: thresholds['global'] ← BEST_THRESHOLD(scores, labels)

27: unique_relations ← UNIQUE(relations)
28: for each r in unique_relations do
29:     mask_r ← (relations = r)
30:     thresholds[r] ← BEST_THRESHOLD(scores[mask_r], labels[mask_r])
31: end for

32: return thresholds
```

## Algorithm 3: Link Prediction Inference

```
Algorithm EVAL_LINK(D_test, model, heads_filter, tails_filter, filt=True, k_list=[1, 3, 10])
Input:
    D_test: Test triplets (h, r, t)
    model: Trained DirectAU-KG model
    heads_filter: Sparse map (t, r) → valid head entities
    tails_filter: Sparse map (h, r) → valid tail entities
    filt: Whether to apply filtered ranking
Output:
    metrics: {mr, mrr, hit@1, hit@3, hit@10}

1: model ← evaluation mode
2: mr_total ← 0.0
3: mrr_total ← 0.0
4: hits_total ← [0, 0, 0]
5: count ← 0
6: chunk_size ← lp_eval_chunk_size

7: for each batch B = (h_batch, r_batch, t_batch) in D_test do
8:     for each triple (h, r, t) in B do
9:         head_scores ← SCORE_ALL_ENTITIES(model, ?, r, t, chunk_size)
10:        tail_scores ← SCORE_ALL_ENTITIES(model, h, r, ?, chunk_size)

11:        if filt then
12:            if heads_filter contains (t, r) and |heads_filter(t, r)| > 1 then
13:                save head_scores[h]
14:                head_scores ← head_scores + heads_filter(t, r) × 1e30
15:                restore head_scores[h]
16:            end if

17:            if tails_filter contains (h, r) and |tails_filter(h, r)| > 1 then
18:                save tail_scores[t]
19:                tail_scores ← tail_scores + tails_filter(h, r) × 1e30
20:                restore tail_scores[t]
21:            end if
22:        end if

23:        head_metrics ← RANKING_METRICS(head_scores, h, k_list)
24:        tail_metrics ← RANKING_METRICS(tail_scores, t, k_list)

25:        mr_total ← mr_total + head_metrics.mr + tail_metrics.mr
26:        mrr_total ← mrr_total + head_metrics.mrr + tail_metrics.mrr
27:        hits_total ← hits_total + head_metrics.hits + tail_metrics.hits
28:        count ← count + 2
29:    end for
30: end for

31: metrics.mr ← mr_total / count
32: metrics.mrr ← mrr_total / count
33: metrics.hit@k ← hits_total[k] / count for each k in k_list
34: return metrics
```

```
Helper function RANKING_METRICS(scores, target, k_list)
1: sorted_idx ← ARGSORT_ASC(scores)
2: target_rank ← POSITION(sorted_idx, target) + 1
3: hits ← [1 if target_rank ≤ k else 0 for each k in k_list]
4: return {mr: target_rank, mrr: 1 / target_rank, hits: hits}
```

## Algorithm 4: Triple Classification Inference

```
Algorithm EVAL_CLASSIFY(D_test, model, thresholds)
Input:
    D_test: Test quadruples (h, r, t, y) where y ∈ {0, 1}
    model: Trained DirectAU-KG model
    thresholds: Relation-specific thresholds with a global fallback
Output:
    metrics: {accuracy, precision, recall, f1, pr_auc, roc_auc}

1: scores ← BATCH_SCORE(model, D_test)
2: labels ← EXTRACT_LABELS(D_test)
3: relations ← EXTRACT_RELATIONS(D_test)
4: is_dist ← model.is_distance_based
5: predictions ← empty array
6: confidence ← empty array

7: for i = 1 to |D_test| do
8:     r ← relations[i]
9:     thresh ← thresholds.get(r, thresholds['global'])
10:
11:    if is_dist then
12:        predictions[i] ← 1 if scores[i] ≤ thresh else 0
13:        confidence[i] ← -scores[i]
14:    else
15:        predictions[i] ← 1 if scores[i] ≥ thresh else 0
16:        confidence[i] ← scores[i]
17:    end if
18: end for

19: metrics ← CLASSIFICATION_METRICS(predictions, labels, confidence)
20: return metrics
```

## Legend & Notation

- `E_e`: Entity embedding matrix
- `E_r`: Relation embedding matrix
- `M_r`: Relation mask embedding matrix
- `L2_NORMALIZE(x)`: $x / (\|x\|_2 + \epsilon)$
- `⊙`: Element-wise product
- `compose_mode`: Relation composition rule, either `mul` or `add`
- `PDIST`: Pairwise Euclidean distances among unique entities in a mini-batch
- `gamma`: Weight for the uniformity regularizer
- `best_perf`: Best validation MRR observed during training
- `FILTER_RANKING_PENALTY`: Large constant (`1e30`) used for filtered link prediction
- `is_distance_based`: True for DirectAU-KG, so smaller scores mean more likely positives
- `thresholds['global']`: Fallback threshold for unseen relations during classification