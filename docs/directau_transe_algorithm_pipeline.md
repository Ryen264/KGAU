# DirectAU - Upgraded Algorithms

This document summarizes the upgraded training and inference algorithms used by the DirectAU-TransE implementation.

## Algorithm 1 — Upgraded Training (DirectAU-TransE)

Inputs: training set D_train, config keys (`model_type`, `gamma_uni`, `gamma_neg`, `epsilon`)

High level steps per minibatch:

- L2-normalize embeddings for `h`, `r`, `t`.
- Apply relation attention mask: `h_mask = normalize(h ⊙ sigmoid(W_r[r]))`.
- Compose query: `q = normalize(h_mask + r)`.
- Alignment loss: mean over batch of ||q - t||_2^2.
- Uniformity loss: for deduplicated entity ids U in the batch:
    - If |U| < 2 then L_uni = 0.0.
    - Else L_uni = log(mean(exp(-2 * PDIST(normalize(E[U]))^2)) + ε).
- Optional negative loss L_neg (used if `gamma_neg > 0`): compute using corrupted candidates
    produced by the `corrupter` (handles single or multiple negatives per sample).
- Final batch loss: L = L_align + gamma_uni * L_uni + gamma_neg * L_neg.

Notes:
- All embeddings are constrained to unit length before composition and scoring.
- `epsilon` (ε) is a small stabilizer added inside logs to avoid numerical issues.

## Algorithm 2 — Inference (Dot-Product Scoring)

Scoring function:

Score(q, c): assert both are normalized, return dot(q, c) — higher is better.

Tasks:

- Link prediction: for head or tail ranking, compute `q` for each candidate entity and
    score by dot product against the candidate's normalized embedding. Filtering is applied
    by subtracting a large penalty for known corrupted entities (keeping the true entity's score).
- Triple classification: compute the dot-product score and compare with relation-specific
    or global thresholds (prediction = 1 if score >= threshold).

## Algorithm 3 — Thresholding (Triple Classification)

- Compute batch scores on validation set using dot-product scoring.
- For global and relation-specific groups, choose the threshold that maximizes
    accuracy on the validation set; for dot-product scoring larger scores indicate
    more likely positive.

## Notation

- `E`, `E_r`, `W_r`: entity, relation, and relation-attention embeddings.
- `L2_NORMALIZE(x)`: normalization to unit-norm.
- `⊙`: element-wise product.
- `PDIST`: pairwise Euclidean distances.
- `gamma_uni`: uniformity loss weight.
- `gamma_neg`: negative-sample loss weight (optional).
- `epsilon` (ε): small stabilizer added inside `log(... + ε)`.

This updated document aligns the implementation with the training and inference changes (normalized composition, uniformity term with ε, optional negative loss, and dot-product inference).
