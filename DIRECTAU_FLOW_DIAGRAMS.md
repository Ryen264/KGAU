# DirectAU-TransE Algorithm Flow Diagrams

## 1. Model Architecture Flow

```
Triples: (h, r, t)
      |
      v
+-----------------------+
|  Entity Embeddings E  |
+-----------------------+
      |
      v
+-----------------------+      +------------------------+
|  Relation Embeddings R|      |  Relation Attn W_r     |
+-----------------------+      +------------------------+
      |                               |
      v                               v
  L2 Normalize                    Sigmoid
      |                               |
      +---------------+---------------+
                      v
             h_mask = normalize(h ⊙ sigmoid(W_r))
                      |
                      v
             q = normalize(h_mask + normalize(r))
                      |
                      v
            +------------------------+
            | Dot-Product Scoring    |
            | score = q · t          |
            +------------------------+
                      |
                      v
             +-----------------------+
             | Loss Computation      |
             |  - Alignment          |
             |  - Uniformity         |
             |  - Optional Negative  |
             +-----------------------+
                      |
                      v
             +-----------------------+
             | Backprop + Optimizer  |
             +-----------------------+
```

## 2. Training Loop Flow

```
START EPOCH
   |
   v
[Shuffle training triples]
   |
   v
FOR each batch (h, r, t):
   |
   +--> Encode & Compose Query q
   |
   +--> Alignment loss: mean(||q - t||^2)
   |
   +--> Uniformity loss over unique entities in batch
   |
   +--> Optional negative loss (if gamma_neg > 0)
   |
   +--> Total loss = L_align + gamma_uni*L_uni + gamma_neg*L_neg
   |
   +--> Backward + optimizer step
   |
   v
END BATCH
   |
   v
[Evaluate every epoch_per_test epochs]
   |
   v
[Early stopping on validation MRR]
   |
   v
END EPOCH
```

## 3. Loss Computation Flow

### Alignment Loss
```
Inputs: q [B x d], t [B x d]
Step 1: diff = q - t
Step 2: dist_sq = ||diff||^2
Step 3: L_align = mean(dist_sq)
```

### Uniformity Loss (Entities)
```
Inputs: unique entity IDs U
Step 1: e = normalize(E[U])
Step 2: dist_sq = pdist(e)^2
Step 3: weights = exp(-2 * dist_sq)
Step 4: L_uni = log(mean(weights) + epsilon)
```

### Optional Negative Loss
```
Inputs: q, corrupted tail IDs t_neg
Step 1: t_neg = normalize(E[t_neg])
Step 2: dist_sq = ||q - t_neg||^2
Step 3: L_neg = log(mean(exp(-2 * dist_sq)) + epsilon)
```

## 4. Inference (Link Prediction)

```
Given test triple (h, r, t)

Tail prediction (h, r, ?):
  q = compose(h, r)
  scores = q · E_all
  rank of t gives tail metrics

Head prediction (?, r, t):
  score all candidate heads via q = compose(head, r)
  rank of h gives head metrics

Filter known positives by adding a large penalty
Aggregate MR, MRR, Hits@K
```

## 5. Triple Classification (Thresholding)

```
Validation:
  score = q · t
  choose best threshold per relation + global threshold

Test:
  predict positive if score >= threshold
  compute Acc, Prec, Rec, F1, PR-AUC, ROC-AUC
```
