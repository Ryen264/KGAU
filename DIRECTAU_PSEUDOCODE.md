# DirectAU-TransE: Complete Pseudocode Reference

## Table of Contents
1. [Initialization](#initialization)
2. [Training](#training)
3. [Validation](#validation)
4. [Testing/Inference](#testing)
5. [Loss Functions](#loss-functions)

---

## Initialization

```python
# ============================================================================
# FUNCTION: Initialize DirectAU-TransE Model
# ============================================================================

FUNCTION initialize_model(config, n_entity, n_relation):

    # Create embeddings
    entity_embed = Embedding(n_entity, dim=config.dim)
    relation_embed = Embedding(n_relation, dim=config.dim)
    relation_attn = Embedding(n_relation, dim=config.dim)

    # Optimizer
    optimizer = Optimizer(
        params=all_trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    RETURN {
        'entity_embed': entity_embed,
        'relation_embed': relation_embed,
        'relation_attn': relation_attn,
        'optimizer': optimizer,
        'gamma_uni': config.gamma,
        'gamma_neg': config.gamma_neg,
        'epsilon': config.epsilon,
    }

END FUNCTION
```

---

## Training

```python
# ============================================================================
# FUNCTION: Compose Query
# ============================================================================

FUNCTION compose_query(h_ids, r_ids, E, R, W):
    h = L2_NORMALIZE(E[h_ids])
    r = L2_NORMALIZE(R[r_ids])
    w = W[r_ids]

    h_mask = L2_NORMALIZE(h * SIGMOID(w))
    q = L2_NORMALIZE(h_mask + r)
    RETURN q
END FUNCTION


# ============================================================================
# FUNCTION: Alignment Loss
# ============================================================================

FUNCTION alignment_loss(q, t):
    diff = q - t
    return MEAN(NORM(diff, p=2)^2)
END FUNCTION


# ============================================================================
# FUNCTION: Uniformity Loss (Entity Embeddings)
# ============================================================================

FUNCTION uniformity_loss(unique_entity_ids, E, epsilon):
    IF len(unique_entity_ids) < 2:
        RETURN 0.0

    e = L2_NORMALIZE(E[unique_entity_ids])
    dist_sq = PDIST(e)^2
    return LOG(MEAN(EXP(-2 * dist_sq)) + epsilon)
END FUNCTION


# ============================================================================
# FUNCTION: Optional Negative Loss
# ============================================================================

FUNCTION negative_loss(q, t_neg_ids, E, epsilon):
    t_neg = L2_NORMALIZE(E[t_neg_ids])
    dist_sq = NORM(q - t_neg, p=2)^2
    return LOG(MEAN(EXP(-2 * dist_sq)) + epsilon)
END FUNCTION


# ============================================================================
# FUNCTION: Train One Epoch
# ============================================================================

FUNCTION train_one_epoch(train_triples, model, corrupter, config):

    SHUFFLE(train_triples)

    FOR each batch (h, r, t):

        q = compose_query(h, r, E, R, W)
        t_emb = L2_NORMALIZE(E[t])

        # Alignment
        L_align = alignment_loss(q, t_emb)

        # Uniformity over unique entities
        unique_entities = UNIQUE(CONCAT(h, t))
        L_uni = uniformity_loss(unique_entities, E, config.epsilon)

        # Optional negative loss
        L_neg = 0.0
        IF config.gamma_neg > 0 AND corrupter is not None:
            t_neg = corrupter.corrupt(h, r, t)
            L_neg = negative_loss(q, t_neg, E, config.epsilon)

        # Total loss
        L = L_align + config.gamma * L_uni + config.gamma_neg * L_neg

        BACKWARD(L)
        OPTIMIZER_STEP()

    RETURN avg_epoch_loss

END FUNCTION
```

---

## Validation

```python
# ============================================================================
# FUNCTION: Validation Link Prediction
# ============================================================================

FUNCTION validate(model, valid_triples, filter_sets):
    metrics = link_prediction(model, valid_triples, filter_sets)
    RETURN metrics['mrr']
END FUNCTION
```

---

## Testing

```python
# ============================================================================
# FUNCTION: Link Prediction
# ============================================================================

FUNCTION link_prediction(model, test_triples, filter_sets):

    FOR each test batch (h, r, t):

        # Tail prediction
        q_tail = compose_query(h, r)
        scores_tail = DOT(q_tail, E_all.T)

        # Head prediction
        scores_head = DOT(compose_query(E_all, r), t)  # conceptual

        # Apply filtering (mask known positives)
        APPLY_FILTER(scores_tail, scores_head, filter_sets)

        # Compute MR, MRR, Hits@K
        UPDATE_METRICS()

    RETURN metrics

END FUNCTION


# ============================================================================
# FUNCTION: Triple Classification (Thresholding)
# ============================================================================

FUNCTION triple_classification(valid_set, test_set):
    # Validation
    thresholds = find_best_thresholds(valid_set)

    # Test
    preds = (scores >= thresholds[relation] or thresholds['global'])
    return classification_metrics(preds, labels)

END FUNCTION
```

---

## Loss Functions

```python
LOSS FUNCTION alignment_loss:
    L_align = mean(||q - t||_2^2)

LOSS FUNCTION uniformity_loss:
    L_uni = log(mean(exp(-2 * ||e_i - e_j||_2^2)) + epsilon)

LOSS FUNCTION negative_loss (optional):
    L_neg = log(mean(exp(-2 * ||q - t_neg||_2^2)) + epsilon)

LOSS FUNCTION total_loss:
    L = L_align + gamma_uni * L_uni + gamma_neg * L_neg
```
