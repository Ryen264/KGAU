# DirectAU-TransE Algorithm (Embedding-Based)

## Overview

**DirectAU-TransE** is a DirectAU-style Knowledge Graph Embedding model that replaces text encoders with learned entity and relation embeddings. It avoids negative sampling by combining:

1. **Alignment loss** to bring each (head, relation) query close to its true tail.
2. **Uniformity loss** to spread entity embeddings on the unit hypersphere.
3. *(Optional)* **Negative loss** for corrupted tails if `gamma_neg > 0`.

This repository trains **DirectAU-TransE** and a **TransE baseline** side-by-side in `main.py`.

---

## Architecture

### Components

- **Entity embedding** $E \in \mathbb{R}^{n_e \times d}$
- **Relation embedding** $R \in \mathbb{R}^{n_r \times d}$
- **Relation attention** $W \in \mathbb{R}^{n_r \times d}$
- **Relation attention bias** $b \in \mathbb{R}^{n_r \times d}$

### Query Composition

Given a triple $(h, r, t)$, the model builds a query vector:

1. Normalize embeddings:
   - $\hat{h} = \text{L2Norm}(E_h)$
   - $\hat{r} = \text{L2Norm}(R_r)$
2. Relation attention mask:
    - $m_r = \sigma(W_r \odot \hat{r} + b_r)$
    - $\tilde{h} = \hat{h} \odot m_r$
3. Compose and normalize:
    - $q = \text{L2Norm}(\tilde{h} + \hat{r})$

### Scoring

- **Dot-product scoring** (higher is better):

$$
\text{score}(h, r, t) = q \cdot \hat{t}
$$

where $\hat{t} = \text{L2Norm}(E_t)$.

---

## Loss Functions

### 1. Alignment Loss

$$
L_{\text{align}} = \frac{1}{B} \sum_{i=1}^{B} \|q_i - \hat{t}_i\|_2^2
$$

- Encourages correct tails to be close to composed queries.

### 2. Uniformity Loss (Entities)

Let $U$ be the unique entity IDs in the batch (heads and tails).

$$
L_{\text{uni}} = \log\left(\mathbb{E}_{i<j}\left[\exp\left(-2\|\hat{e}_i - \hat{e}_j\|_2^2\right)\right]\right)
$$

- Uses pairwise distances over normalized entity embeddings.
- Encourages global spread on the hypersphere.

### 3. Optional Negative Loss

If `gamma_neg > 0`, corrupted tails are generated and used in a DirectAU-style repulsion term:

$$
L_{\text{neg}} = \log\left(\mathbb{E}\left[\exp\left(-2\|q - \hat{t}_{\text{neg}}\|_2^2\right)\right]\right)
$$

### 4. Total Loss

$$
L = L_{\text{align}} + \gamma_{\text{uni}} \cdot L_{\text{uni}} + \gamma_{\text{neg}} \cdot L_{\text{neg}}
$$

---

## Training Algorithm

```
for epoch in range(n_epoch):
    shuffle(train_triples)

    for batch in batches:
        # 1) compose queries
        q = normalize( normalize(h) * sigmoid(W_r ⊙ normalize(r) + b_r) + normalize(r) )

        # 2) alignment loss
        L_align = mean(||q - t||^2)

        # 3) uniformity loss on unique entities in batch
        L_uni = log(mean(exp(-2 * pdist(E_unique)^2)))

        # 4) optional negative loss
        if gamma_neg > 0:
            L_neg = log(mean(exp(-2 * ||q - t_neg||^2)))
        else:
            L_neg = 0

        # 5) total loss
        L = L_align + gamma_uni * L_uni + gamma_neg * L_neg

        # 6) backward + step
        L.backward(); optimizer.step()
```

Evaluation is run every `epoch_per_test` epochs. Early stopping uses validation MRR.

---

## Inference / Testing

### Link Prediction

For each test triple $(h, r, t)$:

1. **Head prediction**: score all entities $e$ for $(e, r, t)$.
2. **Tail prediction**: score all entities $e$ for $(h, r, e)$.
3. **Ranking**: higher dot-product score ranks higher.
4. **Filtering**: known true triples from train/valid/test are masked by a large penalty.
5. **Metrics**: MR, MRR, Hits@K.

### Triple Classification

- Use validation data with labels to find best per-relation and global thresholds.
- Predict positive if score $\ge$ threshold.
- Report Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC.

---

## Configuration Notes

Key config parameters used in this codebase:

- `dim`: embedding dimension
- `gamma`: uniformity weight (mapped to `gamma_uni`)
- `gamma_neg`: optional negative loss weight
- `epsilon`: stabilizer for `log( ... + epsilon )`
- `batch_size`, `n_epoch`, `epoch_per_test`
- `optimizer`, `learning_rate`

---

## Baseline: TransE

TransE is trained with margin ranking loss and Bernoulli negative sampling:

$$
L_{\text{pair}} = \max(0, d(h, r, t) - d(h', r, t') + \text{margin})
$$

This baseline uses the same evaluation pipeline (link prediction and classification).
