# DirectAU-TransE: Algorithm Pipeline

## Algorithm 1: Training Stage

```
Algorithm TRAIN(D_train, n_entity, n_relation, config)
Input:
    D_train: Training triplets (h, r, t)
    n_entity: Number of entities
    n_relation: Number of relations
    config: Configuration (dim, learning_rate, n_epoch, batch_size, gamma, epoch_per_test, early_stop_patience)
Output:
    θ: Trained model parameters
    best_perf: Best validation MRR
    best_epoch: Epoch with best validation performance

1: Initialize entity embeddings E ∈ R^(n_entity × dim)
2: Initialize relation embeddings E_r ∈ R^(n_relation × dim)
3: Initialize relation attention embeddings W_r ∈ R^(n_relation × dim)
4: Initialize parameters uniformly in [-init_range, init_range]
5: Initialize optimizer ← Adam(θ, learning_rate)
6: best_perf ← 0.0
7: patience_count ← 0

8: for epoch = 1 to n_epoch do
9:     total_loss ← 0.0
10:    Shuffle D_train
11:
12:    for each batch B = (h_batch, r_batch, t_batch) in D_train do
13:        batch_size ← |B|
14:        zero gradients

15:        // L2-normalize raw embeddings
16:        h ← L2_NORMALIZE(E[h_batch])
17:        r ← L2_NORMALIZE(E_r[r_batch])
18:        t ← L2_NORMALIZE(E[t_batch])
19:        w ← W_r[r_batch]
20:
21:        // Attention mask on head, then additive composition
22:        h_masked ← L2_NORMALIZE(h ⊙ SIGMOID(w))
23:        q ← L2_NORMALIZE(h_masked + r)

24:        // Alignment loss: squared L2 distance per triple
25:        L_align_per_triple ← ||q - t||_2^2
26:        L_align_total ← SUM(L_align_per_triple)

27:        // Safe uniformity loss over unique entities in the batch
28:        unique_entities ← UNIQUE(h_batch ∪ t_batch)
29:        if |unique_entities| < 2 then
30:            L_uni ← 0.0
31:        else
32:            e_uni ← L2_NORMALIZE(E[unique_entities])
33:            pairwise_dist ← PDIST(e_uni, p=2)
34:            L_uni ← log(MEAN(exp(-2 * pairwise_dist^2)) + ε)  // ε small stabilizer
35:        end if

36:        // Combine losses and update
37:        L_align_norm ← L_align_total / batch_size
38:        L ← L_align_norm + gamma * L_uni
39:        backpropagate L
40:        update θ using optimizer

41:        total_loss ← total_loss + L * batch_size
42:    end for

43:    avg_loss ← total_loss / |D_train|
44:    LOG('Epoch', epoch, 'Loss', avg_loss)

45:    // Validation & early stopping (every epoch_per_test)
46:    if epoch mod epoch_per_test == 0 then
47:        valid_mrr ← EVAL_LINK(D_valid, model)
48:        if valid_mrr > best_perf then
49:            best_perf ← valid_mrr
50:            best_epoch ← epoch
51:            best_state ← COPY(θ)
52:            patience_count ← 0
53:        else
54:            patience_count ← patience_count + 1
55:        end if

56:        if early_stop_patience > 0 AND patience_count ≥ early_stop_patience then
57:            BREAK
58:        end if
59:    end if
60: end for

61: if best_state exists then
62:     θ ← best_state
63: end if
64: SAVE(θ)
65: return θ, best_perf, best_epoch
```

## Algorithm 2: Validation & Thresholding

```
Algorithm FIND_THRESHOLDS(D_valid, model)
Input:
    D_valid: Validation quadruples (h, r, t, y) where y ∈ {0, 1}
    model: Trained DirectAU-TransE model
Output:
    thresholds: Dictionary {global, r → threshold_r}

1: scores ← BATCH_SCORES(model, D_valid)  // distance scores per triple
2: labels ← EXTRACT_LABELS(D_valid)
3: relations ← EXTRACT_RELATIONS(D_valid)
4: thresholds ← {}
5: is_dist ← model.is_distance_based

6: function FIND_BEST_THRESHOLD(scores_subset, labels_subset):
7:     best_acc ← 0.0
8:     best_thresh ← 0.0
9:     candidates ← UNIQUE(scores_subset)
10:
11:    for each thresh in candidates do
12:        if is_dist then
13:            predictions ← (scores_subset ≤ thresh ? 1 : 0)
14:        else
15:            predictions ← (scores_subset ≥ thresh ? 1 : 0)
16:        end if

17:        acc ← MEAN(predictions == labels_subset)
18:        if acc > best_acc then
19:            best_acc ← acc
20:            best_thresh ← thresh
21:        end if
22:    end for
23:    return best_thresh
24: end function

25: thresholds['global'] ← FIND_BEST_THRESHOLD(scores, labels)

26: unique_relations ← UNIQUE(relations)
27: for each r in unique_relations do
28:     mask_r ← (relations == r)
29:     thresholds[r] ← FIND_BEST_THRESHOLD(scores[mask_r], labels[mask_r])
30: end for

31: return thresholds
```

## Algorithm 3: Link Prediction Inference (chunked)

```
Algorithm EVAL_LINK(D_test, model, heads_filter, tails_filter, filt=True, chunk_size)
Input:
    D_test: Test triplets (h, r, t)
    model: Trained DirectAU-TransE model (distance-based)
    heads_filter: Sparse map (t, r) → known head entities
    tails_filter: Sparse map (h, r) → known tail entities
    chunk_size: Max candidates scored per chunk
Output:
    metrics: {mr, mrr, hit@1, hit@3, hit@10}

1: model.eval()
2: mr_total, mrr_total ← 0.0, 0.0
3: hits_total ← [0, 0, 0]
4: count ← 0

5: for each batch B = (h_batch, r_batch, t_batch) in D_test do
6:     for each (h, r, t) in B do
7:         head_scores ← SCORE_ALL_ENTITIES_CHUNKS(model, ?, r, t, chunk_size)
8:         tail_scores ← SCORE_ALL_ENTITIES_CHUNKS(model, h, r, ?, chunk_size)

9:         if filt then
10:            if (t, r) in heads_filter AND |heads_filter(t, r)| > 1 then
11:                save true ← head_scores[h]
12:                head_scores ← head_scores + heads_filter(t, r) * FILTER_RANKING_PENALTY
13:                head_scores[h] ← true
14:            end if
15:            if (h, r) in tails_filter AND |tails_filter(h, r)| > 1 then
16:                save true ← tail_scores[t]
17:                tail_scores ← tail_scores + tails_filter(h, r) * FILTER_RANKING_PENALTY
18:                tail_scores[t] ← true
19:            end if
20:         end if

21:         head_metrics ← RANKING_METRICS(head_scores, h, ascending=True)
22:         tail_metrics ← RANKING_METRICS(tail_scores, t, ascending=True)

23:         mr_total ← mr_total + head_metrics.mr + tail_metrics.mr
24:         mrr_total ← mrr_total + head_metrics.mrr + tail_metrics.mrr
25:         hits_total ← hits_total + head_metrics.hits + tail_metrics.hits
26:         count ← count + 2
27:     end for
28: end for

29: metrics['mr'] ← mr_total / count
30: metrics['mrr'] ← mrr_total / count
31: for i, k in enumerate([1,3,10]) do
32:     metrics['hit@'+str(k)] ← hits_total[i] / count
33: end for
34: return metrics
```

## Algorithm 4: Triple Classification Inference

```
Algorithm EVAL_CLASSIFY(D_test, model, thresholds)
Input:
    D_test: Test quadruples (h, r, t, y)
    model: Trained DirectAU-TransE model
    thresholds: Relation-specific thresholds plus 'global'
Output:
    metrics: {accuracy, precision, recall, f1, pr_auc, roc_auc}

1: scores ← BATCH_SCORES(model, D_test)
2: labels ← EXTRACT_LABELS(D_test)
3: relations ← EXTRACT_RELATIONS(D_test)
4: is_dist ← model.is_distance_based
5: predictions, confidences ← empty lists

6: for i = 1 to |D_test| do
7:     r ← relations[i]
8:     thresh ← thresholds.get(r, thresholds['global'])
9:     if is_dist then
10:        pred ← (scores[i] ≤ thresh ? 1 : 0)
11:        conf ← -scores[i]
12:     else
13:        pred ← (scores[i] ≥ thresh ? 1 : 0)
14:        conf ← scores[i]
15:     end if
16:     append pred to predictions
17:     append conf to confidences
18: end for

19: metrics ← CLASSIFICATION_METRICS(predictions, labels, confidences)
20: return metrics
```

## Legend & Notation

- `E`, `E_r`, `W_r`: Entity, relation, and relation-attention embeddings
- `L2_NORMALIZE(x)`: $x / (\|x\|_2 + \epsilon)$
- `⊙`: Element-wise product
- `SIGMOID`: Sigmoid activation
- `PDIST`: Pairwise Euclidean distances
- `FILTER_RANKING_PENALTY`: Large constant (e.g., 1e30)
- `is_distance_based`: True → smaller score = more likely positive
- `ε`: Small stabilizer to avoid log(0)
# DirectAU-TransE: Algorithm Pipeline (implementation-aligned)

## Algorithm 1: Training Stage

```
Algorithm TRAIN(D_train, n_entity, n_relation, config)
Input:
    D_train: Training triplets (h, r, t)
    n_entity: Number of entities
    n_relation: Number of relations
    config: Configuration (dim, learning_rate, n_epoch, batch_size, epoch_per_test, gamma, compose_mode, early_stop_patience, epsilon)
Output:
    θ: Trained model parameters
    best_perf: Best validation MRR
    best_epoch: Epoch with best validation performance

1: Initialize entity embeddings E_e ∈ R^(n_entity × dim)
2: Initialize relation embeddings E_r ∈ R^(n_relation × dim)
3: Initialize relation attention embeddings W_r ∈ R^(n_relation × dim)
4: Initialize weights uniformly in [-init_range, init_range] (init_range = 6/√dim by default)
5: Initialize optimizer ← OPTIMIZER(θ, lr)
6: best_perf ← 0.0
7: patience_count ← 0

8: for epoch = 1 to n_epoch do
9:     total_loss ← 0.0
10:    Shuffle D_train
11:
12:    for each batch B in D_train do
13:        h_batch, r_batch, t_batch ← B
14:        batch_size ← |B|
15:        zero gradients
16:
17:        // L2 Normalize raw embeddings
18:        h ← L2_NORMALIZE(E_e[h_batch])
19:        r ← L2_NORMALIZE(E_r[r_batch])
20:        t ← L2_NORMALIZE(E_e[t_batch])
21:        w ← W_r[r_batch]
22:
23:        // Relation attention mask and additive composition
24:        h_masked ← L2_NORMALIZE(h ⊙ SIGMOID(w))
25:        q ← L2_NORMALIZE(h_masked + r)
26:
27:        // Alignment loss: squared L2 distance per triple
28:        L_align ← MEAN(||q - t||_2^2)
29:
30:        // Safe uniformity loss over unique entities in batch
31:        unique_entities ← UNIQUE(h_batch ∪ t_batch)
32:        if |unique_entities| < 2 then
33:            L_uni ← 0.0
34:        else
35:            e_uni ← L2_NORMALIZE(E_e[unique_entities])
36:            pairwise_dist ← PDIST(e_uni, p=2)
37:            L_uni ← log(MEAN(exp(-2 * pairwise_dist^2)) + ε)
38:        end if

39:        // Combined loss and optimization
40:        L ← L_align + gamma * L_uni
41:        backpropagate L
42:        update θ using optimizer

43:        total_loss ← total_loss + L * batch_size
44:    end for

45:    avg_loss ← total_loss / |D_train|
46:    LOG('Epoch', epoch, 'Loss', avg_loss)

47:    if (epoch mod epoch_per_test == 0) then
48:        valid_mrr ← EVAL_LINK(D_valid, model)
49:        if valid_mrr > best_perf then
50:            best_perf ← valid_mrr
51:            best_epoch ← epoch
52:            best_state ← COPY(θ)
53:            patience_count ← 0
54:        else
55:            patience_count ← patience_count + 1
56:        end if
57:
58:        if early_stop_patience > 0 AND patience_count ≥ early_stop_patience then
59:            BREAK
60:        end if
61:    end if
62: end for

63: if best_state exists then
64:     θ ← best_state
65: end if
66: SAVE(θ)
67: return θ, best_perf, best_epoch
```

## Algorithm 2: Validation & Thresholding

```
Algorithm FIND_THRESHOLDS(D_valid, model)
Input:
    D_valid: Validation quadruples (h, r, t, y) where y ∈ {0, 1}
    model: Trained DirectAU-TransE model (distance-based)
Output:
    thresholds: Dictionary {global, r → threshold_r}

1: scores ← BATCH_SCORES(model, D_valid)
2: labels ← EXTRACT_LABELS(D_valid)
3: relations ← EXTRACT_RELATIONS(D_valid)
4: thresholds ← {}
5: is_dist ← model.is_distance_based

6: function FIND_BEST_THRESHOLD(scores_subset, labels_subset):
7:     best_acc ← 0.0
8:     best_thresh ← 0.0
9:     candidate_scores ← UNIQUE(scores_subset)
10:
11:    for each thresh in candidate_scores do
12:        if is_dist then
13:            preds ← (scores_subset ≤ thresh ? 1 : 0)
14:        else
15:            preds ← (scores_subset ≥ thresh ? 1 : 0)
16:        end if
17:
18:        acc ← MEAN(preds == labels_subset)
19:        if acc > best_acc then
20:            best_acc ← acc
21:            best_thresh ← thresh
22:        end if
23:    end for
24:    return best_thresh
25: end function

26: thresholds['global'] ← FIND_BEST_THRESHOLD(scores, labels)
27: for each r in UNIQUE(relations) do
28:     mask_r ← (relations == r)
29:     thresholds[r] ← FIND_BEST_THRESHOLD(scores[mask_r], labels[mask_r])
30: end for

31: return thresholds
```

## Algorithm 3: Link Prediction Inference (chunked)

```
Algorithm EVAL_LINK(D_test, model, heads_filter, tails_filter, filt=True, chunk_size)
Input:
    D_test: Test triplets (h, r, t)
    model: Trained DirectAU-TransE model
    heads_filter: Sparse map (t, r) → valid head entities
    tails_filter: Sparse map (h, r) → valid tail entities
    chunk_size: Maximum candidate entities per chunk
Output:
    metrics: {mr, mrr, hit@1, hit@3, hit@10}

1: set model to eval mode
2: mr_total, mrr_total ← 0.0, 0.0
3: hits_total ← [0,0,0]
4: count ← 0

5: for each batch (h_batch, r_batch, t_batch) in D_test do
6:     for each (h, r, t) in batch do
7:         head_scores ← SCORE_ALL_ENTITIES_CHUNKS(model, ?, r, t, chunk_size)
8:         tail_scores ← SCORE_ALL_ENTITIES_CHUNKS(model, h, r, ?, chunk_size)

9:         if filt then
10:            if (t, r) in heads_filter then
11:                tmp ← head_scores[h]
12:                head_scores ← head_scores + heads_filter(t, r) * FILTER_RANKING_PENALTY
13:                head_scores[h] ← tmp
14:            end if
15:            if (h, r) in tails_filter then
16:                tmp ← tail_scores[t]
17:                tail_scores ← tail_scores + tails_filter(h, r) * FILTER_RANKING_PENALTY
18:                tail_scores[t] ← tmp
19:            end if
20:         end if

21:         head_metrics ← RANKING_METRICS(head_scores, h, ascending=True)
22:         tail_metrics ← RANKING_METRICS(tail_scores, t, ascending=True)

23:         mr_total += head_metrics.mr + tail_metrics.mr
24:         mrr_total += head_metrics.mrr + tail_metrics.mrr
25:         hits_total = hits_total + head_metrics.hits + tail_metrics.hits
26:         count += 2
27:     end for
28: end for

29: metrics['mr'] ← mr_total / count
30: metrics['mrr'] ← mrr_total / count
31: for i,k in enumerate([1,3,10]) do
32:     metrics['hit@' + k] ← hits_total[i] / count
33: end for
34: return metrics
```

## Algorithm 4: Triple Classification Inference

```
Algorithm EVAL_CLASSIFY(D_test, model, thresholds)
Input:
    D_test: Test quadruples (h, r, t, y)
    model: Trained DirectAU-TransE model
    thresholds: Relation-specific thresholds with 'global' fallback
Output:
    metrics: {accuracy, precision, recall, f1, pr_auc, roc_auc}

1: scores ← BATCH_SCORES(model, D_test)
2: labels ← EXTRACT_LABELS(D_test)
3: relations ← EXTRACT_RELATIONS(D_test)
4: is_dist ← model.is_distance_based
5: predictions, conf_scores ← empty lists

6: for i = 1 to |D_test| do
7:     r ← relations[i]
8:     thresh ← thresholds.get(r, thresholds['global'])
9:     if is_dist then
10:        pred ← (scores[i] ≤ thresh ? 1 : 0)
11:        conf ← -scores[i]
12:     else
13:        pred ← (scores[i] ≥ thresh ? 1 : 0)
14:        conf ← scores[i]
15:     end if
16:     append pred to predictions
17:     append conf to conf_scores
18: end for

19: metrics ← CLASSIFICATION_METRICS(predictions, labels, conf_scores)
20: return metrics
```

## Legend & Notation

- `E_e`, `E_r`, `W_r`: Entity, relation, and relation-attention embeddings
- `L2_NORMALIZE(x)`: $x / (\|x\|_2 + \epsilon)$
- `⊙`: Element-wise product
- `SIGMOID`: Sigmoid activation
- `PDIST`: Pairwise Euclidean distances among unique entities
- `FILTER_RANKING_PENALTY`: Large constant (e.g. `1e30`) for filtered ranking
- `is_distance_based`: True for this model; lower score → more likely positive
