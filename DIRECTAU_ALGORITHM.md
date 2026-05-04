# DirectAU-KG Algorithm: With DistilBERT Encoder

## Overview

**DirectAU-KG** is a Knowledge Graph Embedding model that performs **link prediction without negative sampling**. It uses a bi-encoder architecture with DistilBERT (or other pre-trained transformers) to encode queries (head+relation pairs) and candidate entities, then learns via two complementary losses:

1. **Alignment Loss**: Brings matching triples close together in embedding space
2. **Uniformity Loss**: Prevents collapse by spreading different queries/entities uniformly across the hypersphere

---

## Architecture

### Components

```
Input Knowledge Graph
        ↓
    ┌─────────────────────────────────────┐
    │   Entity/Relation Text Descriptions │
    └─────────────────────────────────────┘
        ↓         ↓
    [HR Encoder]  [T Encoder]  ← DistilBERT pre-trained
        ↓         ↓
   L2 Normalize  L2 Normalize
        ↓         ↓
    Query Embeddings    Tail Entity Embeddings
        ↓         ↓
    ───────────────────
        Ranking/Scoring
```

### Encoder Towers

- **HR Encoder**: Encodes text pairs (head + relation) to query embeddings
  - Input: Entity description text + Relation description text
  - Output: Query embedding vector (normalized to unit hypersphere)

- **T Encoder**: Encodes entity descriptions to tail embeddings
  - Input: Entity description text
  - Output: Tail embedding vector (normalized to unit hypersphere)

Both encoders share the same DistilBERT backbone structure but have independent weights, allowing specialized representation learning for queries vs. entities.

### Key Settings (DistilBERT Configuration)

From `config_wn18rr_distilbert-base-uncased.yaml`:

```yaml
encoder_name: distilbert-base-uncased
dim: 384                           # Hidden dimension of DistilBERT
max_length: 64                      # Max tokens per description
encode_batch_size: 14               # Micro-batch for encoding
gradient_checkpointing: false       # Memory optimization
freeze_lower_layers: 0              # Train all layers
uniformity_max_samples: 170         # Subsampling for uniformity
uniformity_chunk_size: 128          # Chunking to avoid OOM
forward_chunk_size: 32768           # Chunking for forward pass
```

---

## Loss Functions

### 1. Alignment Loss (ALIGN)

**Purpose**: Ensure matching (h, r, t) triples map close together in embedding space.

**Formula**:
$$L_{\text{align}} = \text{mean}\left(\| q - t \|_2^2\right)$$

Where:
- $q = \text{encode\_query}(h, r)$ (query embedding)
- $t = \text{encode\_tail}(t)$ (tail embedding)
- Both are L2-normalized: $\|q\|_2 = \|t\|_2 = 1$

**Computation**:
```python
q_full = self.model.encode_query(h_batch, r_batch)      # [batch_size, 384]
t_full = self.model.encode_tail(t_batch)                # [batch_size, 384]
loss_align = (q_full - t_full).norm(p=2, dim=-1).pow(2).mean()
```

**Intuition**: For correct triples, the (head, relation) query should produce an embedding very close to the tail embedding, minimizing squared Euclidean distance on the unit hypersphere.

---

### 2. Uniformity Loss (UNI)

**Purpose**: Prevent representation collapse; spread different queries and entities uniformly across the hypersphere.

**Formula**:
$$L_{\text{uni}}(x) = \log\left(\text{mean}_{i<j}\left[\exp\left(-2 \| x_i - x_j \|_2^2\right)\right]\right)$$

**High-level idea**:
- Compute pairwise distances between all query/entity embeddings in a batch
- Downweight pairs that are far apart (via exponential decay)
- Maximize the log-mean of these weights → spreads embeddings apart

**Chunked Implementation** (handles large batches efficiently):
```python
def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
    # Input: x of shape [n_samples, 384]
    
    # Optional subsampling (for memory)
    if self.uniformity_max_samples > 0 and x.size(0) > self.uniformity_max_samples:
        idx = torch.randperm(x.size(0))[:self.uniformity_max_samples]
        x = x[idx]
    
    n = x.size(0)
    pair_sum = 0.0
    pair_count = 0
    
    # Iterate over chunks to avoid O(n²) memory
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        xi = x[i_start:i_end]
        
        for j_start in range(i_start, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            xj = x[j_start:j_end]
            
            # Pairwise squared distances
            dist_sq = (xi.unsqueeze(1) - xj.unsqueeze(0)).pow(2).sum(dim=-1)
            weights = torch.exp(-2 * dist_sq)
            
            # Skip diagonal (same embeddings)
            if i_start == j_start:
                diag_mask = torch.eye(i_end - i_start, dtype=torch.bool)
                valid = weights.masked_select(~diag_mask)
                pair_sum += valid.sum() * 0.5
                pair_count += valid.numel() // 2
            else:
                pair_sum += weights.sum()
                pair_count += weights.numel()
    
    return torch.log(pair_sum / pair_count)
```

**Two Uniformity Terms**:
- Computed separately for unique **queries** and unique **entities** within a batch
- Combined as: $L_{\text{uni}} = 0.5 \times (L_{\text{uni}}^q + L_{\text{uni}}^t)$

---

### 3. Total Loss

**Formula**:
$$L_{\text{total}} = L_{\text{align}} + \gamma \cdot L_{\text{uni}}$$

Where $\gamma$ is a hyperparameter (typically 1.0) controlling the uniformity weight.

---

## Training Algorithm

### Data Preparation

1. **Index entities and relations** from train/valid/test files
2. **Load text descriptions** for each entity and relation (e.g., from WordNet definitions)
3. **Tokenize all descriptions** (cached or on-the-fly)

### Training Loop

```
for epoch in range(n_epochs):
    
    # Shuffle training data
    rand_idx = torch.randperm(n_train)
    heads, relations, tails ← reorder by rand_idx
    
    opt.zero_grad()
    epoch_loss = 0.0
    
    for batch_idx, (h_batch, r_batch, t_batch) in enumerate(batches):
        
        # Forward pass with automatic mixed precision (AMP)
        with torch.autocast(dtype=fp16):
            
            # 1. Encode queries and tails
            q_batch = encode_query(h_batch, r_batch)          # Query embeddings
            t_batch = encode_tail(t_batch)                    # Tail embeddings
            
            # 2. Compute alignment loss
            loss_align = ||q_batch - t_batch||₂²  (mean)
            
            # 3. Extract unique queries and entities from batch
            unique_queries = torch.unique(stack([h_batch, r_batch], dim=1), dim=0)
            unique_tails = torch.unique(t_batch)
            
            q_unique_emb = encode_query(unique_queries[:, 0], unique_queries[:, 1])
            t_unique_emb = encode_tail(unique_tails)
            
            # 4. Compute uniformity loss
            loss_uni_q = uniformity_loss(q_unique_emb)
            loss_uni_t = uniformity_loss(t_unique_emb)
            loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
            
            # 5. Total loss
            loss = loss_align + gamma * loss_uni
        
        # Backward with gradient accumulation
        loss_scaled = loss / grad_accum_steps
        scaler.scale(loss_scaled).backward()
        
        # Optimizer step every grad_accum_steps batches
        if (batch_idx % grad_accum_steps == 0):
            scaler.step(optimizer)
            scaler.update()
            opt.zero_grad()
    
    # Flush remaining gradients
    if batch_idx % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
    
    # Validation
    valid_perf = validate(valid_data)
    
    if valid_perf > best_valid_perf:
        save_checkpoint()
        best_valid_perf = valid_perf
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        break
```

### Key Training Features

| Feature | Benefit |
|---------|---------|
| **Gradient Accumulation** | Train on larger effective batches with limited GPU memory |
| **Automatic Mixed Precision (AMP)** | Speed up training, reduce memory via FP16 computation |
| **Gradient Checkpointing** | Trade compute for memory by recomputing activations during backward |
| **Freeze Lower Layers** | Stabilize pre-trained DistilBERT weights, fine-tune upper layers |
| **Chunked Encoding** | Encode large batches in micro-batches to prevent OOM |
| **Chunked Uniformity** | Compute uniformity loss on subsampled/chunked data |

---

## Inference / Testing Algorithm

### Link Prediction Task: (?, r, t) and (h, r, ?)

For each test triple (h, r, t):

```
1. Pre-compute: Encode ALL entity descriptions once
   entity_matrix = [encode_tail(0), encode_tail(1), ..., encode_tail(n_entity-1)]
   Shape: [n_entity, 384]

2. For tail prediction (h, r, ?):
   q_tail = encode_query(h, r)                         # [384]
   tail_scores = -dot(q_tail, entity_matrix.T)         # [n_entity]
   rank_tail = argsort(tail_scores)
   target_rank = rank_tail.index(t) + 1
   
3. For head prediction (?, r, t):
   inv_relation_text = "inverse " + relation_text
   q_head = encode_query(t, inv_relation_text)         # Use inverse relation
   head_scores = -dot(q_head, entity_matrix.T)         # [n_entity]
   rank_head = argsort(head_scores)
   target_rank = rank_head.index(h) + 1

4. Apply filtering (optional):
   - Get set of valid candidates S from training data
   - For entities not in S, add penalty to scores
   - Recompute ranks with penalty

5. Compute metrics:
   MR = average target rank
   MRR = average 1/target_rank
   Hits@K = % triples ranked in top K
```

**Code snippet**:
```python
def test_link(self, test_data):
    entity_matrix = self.model.encode_tail(torch.arange(n_entity))
    
    for head, relation, tail in test_data:
        # Tail prediction
        q_tail = self.model.encode_query(head, relation)
        tail_scores = -(q_tail @ entity_matrix.T)
        
        # Head prediction (inverse)
        inv_rel_texts = ["inverse " + rel_text]
        q_head = self.model.encode_query_with_relation_texts(tail, inv_rel_texts)
        head_scores = -(q_head @ entity_matrix.T)
        
        # Rank & filter
        tail_metrics = ranking_metrics(tail_scores, tail_id)
        head_metrics = ranking_metrics(head_scores, head_id)
```

### Why No Negative Sampling?

Traditional KG models (TransE, DistMult) use **negative sampling**:
- Sample corrupted triples: (h', r, t) or (h, r, t') where h'≠h or t'≠t
- Compute pairwise ranking loss: $L = \max(0, \gamma + d(\text{good}) - d(\text{bad}))$

**DirectAU advantages**:
- ✅ **No negative sampling needed**: Uses alignment + uniformity instead
- ✅ **Text encoders**: Leverages semantic information from entity/relation descriptions
- ✅ **Pre-trained backbone**: Benefit from large-scale language model pre-training
- ✅ **All-entity ranking**: Direct ranking against all entities (not just sampled negatives)

---

## Memory Optimizations

### 1. Gradient Checkpointing
```python
self.hr_encoder.gradient_checkpointing_enable()
self.t_encoder.gradient_checkpointing_enable()
```
- Recompute activations during backward pass instead of storing them
- Trade: ~30% faster training but uses ~2-3x more compute

### 2. Automatic Mixed Precision (AMP)
```python
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
    # Forward pass in FP16
    loss = ...
scaler.scale(loss).backward()  # Backward in FP32
```
- ~2x memory reduction
- ~20-30% speedup

### 3. Chunked Encoding
```python
for batch_start in range(0, len(texts), encode_batch_size):
    batch_end = batch_start + encode_batch_size
    encoded = encode(texts[batch_start:batch_end])
```
- Never encode entire dataset at once
- Keeps intermediate activations small

### 4. Uniformity Subsampling
```python
if len(x) > uniformity_max_samples:
    idx = torch.randperm(len(x))[:uniformity_max_samples]
    x = x[idx]
```
- Compute uniformity on subset to avoid O(n²) memory in pairwise distances

---

## Configuration for DistilBERT

### Recommended Settings

```yaml
DirectAUKG:
  encoder_name: distilbert-base-uncased          # Pre-trained transformer
  max_length: 64                                  # Token limit per description
  encode_batch_size: 14                           # Micro-batch for encoding
  
  # Training
  optimizer: AdamW                                # AdamW for transformer
  learning_rate: 0.00003                          # Lower LR for fine-tuning
  n_epoch: 10
  n_batch: 512                                    # Mini-batches per epoch
  grad_accum_steps: 1                             # Gradient accumulation
  
  # Loss weights
  gamma: 1.0                                      # Uniformity weight
  
  # Memory
  amp: true                                       # Mixed precision
  amp_dtype: fp16
  gradient_checkpointing: false
  freeze_lower_layers: 0                          # Train all layers
  freeze_embeddings: false
  freeze_tail_encoder: false
  
  # Uniformity computation
  uniformity_max_samples: 170                     # Subsample for uniformity
  uniformity_chunk_size: 128
  forward_chunk_size: 32768
```

---

## Key Differences from Negative Sampling Models

| Aspect | DirectAU-KG | TransE + Neg Sampling |
|--------|------------|----------------------|
| **Loss** | Alignment + Uniformity | Margin ranking |
| **Negative Samples** | None (all-entity ranking) | Bernoulli/uniform corruption |
| **Text Input** | Entity/relation descriptions | Entity/relation IDs only |
| **Encoder** | DistilBERT (pre-trained) | Learnable embeddings |
| **Inference** | Rank against all entities | Rank against all entities |
| **Memory** | Higher (text encoding) | Lower (direct embeddings) |
| **Semantic Richness** | Very high | Low (learned embeddings) |

---

## Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| **Encode texts** | $O(n \times |T| \times d)$ | $O(|T| \times d)$ |
| **Alignment loss** | $O(B \times d)$ | $O(B \times d)$ |
| **Uniformity loss** | $O(n^2)$ or $O(m^2)$ with subsampling | $O(m \times d)$ |
| **Forward pass** | $O(B \times n \times d)$ | $O(B \times d)$ or chunked |
| **Inference (all entities)** | $O(n \times d)$ | $O(n \times d)$ |

Where: $n$ = num entities, $B$ = batch size, $m$ = uniformity_max_samples, $d$ = embedding dim

---

## Summary

**DirectAU-KG** is an end-to-end differentiable Knowledge Graph embedding framework that:

1. **Encodes semantic information** via DistilBERT
2. **Learns without negative sampling** using alignment + uniformity losses
3. **Optimizes for L2-normalized embeddings** on the unit hypersphere
4. **Scales efficiently** through gradient accumulation, AMP, and chunked computation
5. **Performs direct ranking** against all entities for link prediction

The key insight is that **semantic representations** (via text) + **uniformity constraint** can replace explicit negative sampling, leading to simpler, more interpretable training while leveraging pre-trained language models.
