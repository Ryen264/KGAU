# DirectAU-KG Algorithm Flow Diagrams

## 1. Model Architecture Flow

```
Knowledge Graph Triples: (h, r, t)
        ↓
    ┌───────────────────────────────────────┐
    │ Load Entity/Relation Text Descriptions│
    │ (e.g., from WordNet, entity names)    │
    └───────────────────────────────────────┘
        ↓
    ┌──────────────────┐    ┌──────────────────┐
    │ Head + Relation  │    │   Tail Entity    │
    │ Text Pair        │    │   Text           │
    └──────────────────┘    └──────────────────┘
         ↓                       ↓
    [DistilBERT]          [DistilBERT]
    HR Encoder            T Encoder
         ↓                       ↓
    Mean Pooling         Mean Pooling
         ↓                       ↓
    L2 Normalize          L2 Normalize
         ↓                       ↓
    Query Embeddings      Tail Embeddings
    (e.g., 384-dim)       (e.g., 384-dim)
         ↓                       ↓
         └───────────────┬───────┘
                         ↓
            ┌────────────────────────┐
            │  Loss Computation      │
            │  ─────────────────────│
            │  • Alignment Loss     │
            │  • Uniformity Loss    │
            │  • Combined Loss      │
            └────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │   Backpropagation     │
            │   & Optimization      │
            └────────────────────────┘
```

## 2. Training Loop Flow

```
START EPOCH
    ↓
[Shuffle training data]
    ↓
FOR each batch of (h, r, t):
    │
    ├─→ [Encode h, r with HR Encoder]  → q_batch [B × 384]
    │
    ├─→ [Encode t with T Encoder]      → t_batch [B × 384]
    │
    ├─→ [Alignment Loss]
    │   loss_align = mean(||q - t||₂²)
    │
    ├─→ [Extract Unique Queries]
    │   unique_q = unique([h, r] pairs)
    │   q_unique_emb = encode(unique_q)
    │
    ├─→ [Extract Unique Tails]
    │   unique_t = unique(t)
    │   t_unique_emb = encode(unique_t)
    │
    ├─→ [Uniformity Loss]
    │   loss_uni_q = uniformity(q_unique_emb)
    │   loss_uni_t = uniformity(t_unique_emb)
    │   loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
    │
    ├─→ [Total Loss]
    │   loss = loss_align + gamma * loss_uni
    │
    ├─→ [Scale by Gradient Accumulation]
    │   loss_scaled = loss / grad_accum_steps
    │
    ├─→ [Backward Pass]
    │   loss_scaled.backward()
    │
    └─→ [Optimizer Step (every grad_accum_steps)]
         if batch_idx % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    ↓
[END BATCH]
    ↓
[Validation]
    ├─→ IF valid_perf > best_valid_perf:
    │     [Save Checkpoint]
    │     patience_counter = 0
    └─→ ELSE:
         patience_counter += 1
    ↓
[Early Stopping Check]
    ├─→ IF patience_counter >= patience:
    │     [BREAK]
    └─→ ELSE:
         [Continue to next epoch]
    ↓
END EPOCH
```

## 3. Loss Function Computation

### Alignment Loss
```
Input: Query batch [B × 384], Tail batch [B × 384]
       (both L2-normalized)

Step 1: Compute difference vectors
        diff = q_batch - t_batch  [B × 384]

Step 2: Compute L2 norms
        norms = ||diff||₂        [B]

Step 3: Square the norms
        squared_norms = norms²   [B]

Step 4: Take mean
        loss_align = mean(squared_norms)  scalar

Output: Scalar loss value
```

### Uniformity Loss (Simplified)
```
Input: Unique embeddings x [N × 384]
       (N = num unique queries or entities)

Step 1: Subsample if N > uniformity_max_samples
        if N > max_samples:
            x = x[random_indices[:max_samples]]
        N = min(N, max_samples)

Step 2: Iterate over chunks to compute pairwise distances
        For each chunk i:
            For each chunk j >= i:
                dist_sq[i,j] = ||x_i - x_j||₂²  [chunk_size × chunk_size]
                weights[i,j] = exp(-2 * dist_sq[i,j])
                
                if i == j: exclude diagonal
                pair_sum += sum(weights[i,j] without diag)
                
                else:
                pair_sum += sum(weights[i,j])
                pair_count += weights[i,j].numel()

Step 3: Compute uniformity
        loss_uni = log(pair_sum / pair_count)

Output: Scalar loss value
        (higher = more uniform distribution)

Total uniformity = 0.5 * (loss_uni_q + loss_uni_t)
```

## 4. Inference (Link Prediction)

```
Given: Test triple (h_test, r_test, t_test)

Step 1: Pre-compute all entity embeddings (done once per test set)
        entity_matrix = [encode(e_0), encode(e_1), ..., encode(e_n)]
        Shape: [n_entities × 384]

Step 2: Tail Prediction (h_test, r_test, ?)
        ├─→ q_tail = encode_query(h_test, r_test)
        ├─→ scores = -dot(q_tail, entity_matrix.T)  [n_entities]
        │   (negative dot product for ranking)
        ├─→ sorted_indices = argsort(scores)
        ├─→ target_rank = position of t_test in sorted_indices + 1
        └─→ metrics_tail = {MR, MRR, Hits@K}

Step 3: Head Prediction (?, r_test, t_test)
        ├─→ inv_rel_text = "inverse " + r_test_text
        ├─→ q_head = encode_query(t_test, inv_rel_text)
        ├─→ scores = -dot(q_head, entity_matrix.T)  [n_entities]
        ├─→ sorted_indices = argsort(scores)
        ├─→ target_rank = position of h_test in sorted_indices + 1
        └─→ metrics_head = {MR, MRR, Hits@K}

Step 4: Apply Filtering (if enabled)
        For each prediction direction:
            - Get valid entities from training data
            - For entities NOT in valid set:
                scores[invalid_entities] += PENALTY (e.g., 1e30)
            - Recompute ranks

Step 5: Aggregate Metrics
        MR = (MR_tail + MR_head) / 2
        MRR = (MRR_tail + MRR_head) / 2
        Hits@K = (Hits@K_tail + Hits@K_head) / 2
```

## 5. Forward Pass with Unique Caching

```
Input: Query IDs [B], Relation IDs [B], Tail IDs [B]

Step 1: Stack head + relation pairs
        query_pairs = stack([h, r], dim=1)  [B × 2]

Step 2: Find unique queries
        unique_queries, q_inverse = torch.unique(
            query_pairs, 
            dim=0, 
            return_inverse=True
        )
        # q_inverse maps original indices to unique indices
        # e.g., if queries = [0, 0, 1], then q_inverse = [0, 0, 1]

Step 3: Encode unique queries
        q_unique_emb = encode_query(
            unique_queries[:, 0],  # unique heads
            unique_queries[:, 1]   # unique relations
        )  [num_unique_queries × 384]

Step 4: Map back to original batch size
        q_batch_emb = q_unique_emb[q_inverse]  [B × 384]

Step 5: Find unique tails
        unique_tails, t_inverse = torch.unique(
            tail_ids, 
            return_inverse=True
        )  [num_unique_tails]

Step 6: Encode unique tails
        t_unique_emb = encode_tail(unique_tails)  [num_unique_tails × 384]

Step 7: Map back to original batch size
        t_batch_emb = t_unique_emb[t_inverse]  [B × 384]

Step 8: Compute distances
        dist = ||q_batch_emb - t_batch_emb||₂  [B]

Output: Distance scores for each triple in batch
```

## 6. Memory-Efficient Chunked Forward Pass

```
Input: Large batch size where B × dim might cause OOM

Step 1: Calculate if chunking needed
        if B * dim <= forward_chunk_size:
            [No chunking needed, proceed normally]
        else:
            [Need to chunk the forward pass]

Step 2: Split batch into chunks
        chunk_size = forward_chunk_size // dim
        
        for chunk_start in range(0, B, chunk_size):
            chunk_end = min(chunk_start + chunk_size, B)
            
            ├─→ Get indices for this chunk
            │   q_idx = q_inverse[chunk_start:chunk_end]
            │   t_idx = t_inverse[chunk_start:chunk_end]
            │
            ├─→ Get embeddings for this chunk
            │   q_part = q_unique_emb[q_idx]
            │   t_part = t_unique_emb[t_idx]
            │
            └─→ Compute distances for this chunk
                dist_part = ||q_part - t_part||₂
                dist_parts.append(dist_part)

Step 3: Concatenate all chunks
        final_dist = concatenate(dist_parts)  [B]

Output: Same as non-chunked version, but processed in smaller pieces
```

## 7. Complete Training vs Inference Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                      START EXPERIMENT                       │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA LOADING & PREPARATION                        │
│  • Load entity/relation descriptions                        │
│  • Create ID mappings                                       │
│  • Prepare train/valid/test splits                          │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL INITIALIZATION                              │
│  • Load pre-trained DistilBERT                              │
│  • Create HR & T encoders                                   │
│  • Initialize optimizer (AdamW)                             │
│  • Setup AMP scaler                                         │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING LOOP (Multiple Epochs)                  │
│                                                              │
│  FOR each epoch:                                            │
│   ├─ Shuffle train data                                     │
│   ├─ FOR each batch:                                        │
│   │   ├─ Encode queries & tails                             │
│   │   ├─ Compute alignment + uniformity loss               │
│   │   ├─ Backward & optimize (with grad accum)              │
│   │   └─ Log progress                                       │
│   │                                                          │
│   ├─ Validation on full validation set                      │
│   ├─ IF valid_perf > best:                                  │
│   │   └─ Save checkpoint                                    │
│   ├─ IF early_stop_patience exceeded:                       │
│   │   └─ Break                                              │
│   └─ Test evaluation (every epoch_per_test epochs)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: LOAD BEST MODEL                                   │
│  • Load weights from best validation checkpoint             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: FINAL INFERENCE                                   │
│                                                              │
│  Pre-compute:                                               │
│   • Encode all entities once → entity_matrix               │
│                                                              │
│  FOR each test triple:                                      │
│   ├─ Tail prediction: rank entities for (h, r, ?)          │
│   ├─ Head prediction: rank entities for (?, r, t)          │
│   ├─ Apply filtering (optional)                             │
│   └─ Compute MR, MRR, Hits@K                                │
│                                                              │
│  Final Results:                                             │
│   • Mean MR (average rank)                                  │
│   • Mean MRR (average reciprocal rank)                      │
│   • Hits@1, Hits@3, Hits@10 (% in top-K)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                    END EXPERIMENT                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $h, r, t$ | Head entity ID, Relation ID, Tail entity ID |
| $q$ | Query embedding (from head + relation) |
| $t_{emb}$ | Tail entity embedding |
| $\|\cdot\|_2$ | L2 (Euclidean) norm |
| $\sigma$ | L2 normalization function |
| $\gamma$ | Uniformity loss weight hyperparameter |
| $B$ | Batch size |
| $d$ | Embedding dimension (384 for DistilBERT) |
| $n$ | Number of entities in knowledge graph |

---

## Implementation Notes

1. **Text Encoding Cache**: Consider caching encoded entity/relation embeddings to avoid re-encoding the same descriptions multiple times.

2. **Batch Padding**: When encoding text pairs of different lengths, the tokenizer automatically pads to max_length.

3. **Gradient Accumulation**: Effective batch size = micro_batch_size × grad_accum_steps

4. **Inverse Relations**: Handled implicitly by adding "inverse " prefix to relation text during head prediction encoding.

5. **Filtering**: Sparse tensors store (h,r) → {t} mappings to quickly check which entities are valid for filtering.
