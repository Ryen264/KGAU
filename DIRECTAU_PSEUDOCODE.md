# DirectAU-KG: Complete Pseudocode Reference

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
# FUNCTION: Initialize DirectAUKG Model
# ============================================================================

FUNCTION initialize_model():
    
    # Load configuration
    config = load_config("config_wn18rr_distilbert-base-uncased.yaml")
    
    # Load knowledge graph
    kb_index = index_entities_relations(train_path, valid_path, test_path)
    n_entity, n_relation = graph_size(kb_index)
    
    # Load text descriptions
    entity_texts = load_entity_descriptions()        # [n_entity] list of strings
    relation_texts = load_relation_descriptions()    # [n_relation] list of strings
    
    # Initialize encoders
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name="distilbert-base-uncased"
    )
    
    hr_encoder = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )
    t_encoder = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )
    
    # Move to GPU
    hr_encoder = hr_encoder.to(device)
    t_encoder = t_encoder.to(device)
    
    # Enable memory optimizations
    if gradient_checkpointing:
        hr_encoder.gradient_checkpointing_enable()
        t_encoder.gradient_checkpointing_enable()
    
    if freeze_embeddings:
        freeze_layer(hr_encoder.embeddings)
        freeze_layer(t_encoder.embeddings)
    
    if freeze_lower_layers > 0:
        num_freeze = min(freeze_lower_layers, len(hr_encoder.encoder.layer))
        for i in range(num_freeze):
            freeze_layer(hr_encoder.encoder.layer[i])
            freeze_layer(t_encoder.encoder.layer[i])
    
    # Get trainable parameters
    trainable_params = [p for p in hr_encoder.parameters() if p.requires_grad]
    trainable_params += [p for p in t_encoder.parameters() if p.requires_grad]
    
    # Initialize optimizer
    optimizer = AdamW(
        params=trainable_params,
        lr=learning_rate,          # 3e-5 for DistilBERT
        weight_decay=0.0
    )
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    RETURN {
        'hr_encoder': hr_encoder,
        't_encoder': t_encoder,
        'tokenizer': tokenizer,
        'optimizer': optimizer,
        'scaler': scaler,
        'entity_texts': entity_texts,
        'relation_texts': relation_texts,
        'n_entity': n_entity,
        'n_relation': n_relation,
    }

END FUNCTION
```

---

## Training

```python
# ============================================================================
# FUNCTION: Text Encoding (Tokenize + Pass through Encoder)
# ============================================================================

FUNCTION encode_text_pairs(left_texts, right_texts, encoder, tokenizer, max_length, batch_size):
    """
    Encode text pairs (e.g., "entity_desc" + "relation_desc")
    Returns normalized embeddings on unit hypersphere
    """
    
    all_embeddings = []
    
    FOR batch_start FROM 0 TO len(left_texts) STEP batch_size:
        batch_end = MIN(batch_start + batch_size, len(left_texts))
        left_batch = left_texts[batch_start:batch_end]
        right_batch = right_texts[batch_start:batch_end]
        
        # Tokenize both text inputs together
        encoded = tokenizer(
            left_batch,
            right_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass through encoder
        hidden = encoder(**encoded).last_hidden_state      # [batch_size, seq_len, 768]
        
        # Mean pooling with attention mask
        mask = encoded['attention_mask'].unsqueeze(-1)
        sum_embeddings = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_embeddings / denom                    # [batch_size, 768]
        
        # L2 normalize to unit hypersphere
        normalized = pooled / (pooled.norm(p=2, dim=-1, keepdim=True) + epsilon)
        
        all_embeddings.append(normalized)
    
    RETURN torch.cat(all_embeddings, dim=0)

END FUNCTION


# ============================================================================
# FUNCTION: ID to Text Conversion
# ============================================================================

FUNCTION ids_to_texts(ids, text_table):
    """Convert tensor of IDs to list of text descriptions"""
    RETURN [text_table[i.item()] for i in ids.detach().cpu()]
END FUNCTION


# ============================================================================
# FUNCTION: Encode Query (head + relation pair)
# ============================================================================

FUNCTION encode_query(head_ids, relation_ids, hr_encoder, tokenizer, entity_texts, relation_texts):
    """
    Encode query: (head_entity, relation) -> query_embedding
    """
    
    head_texts = ids_to_texts(head_ids, entity_texts)
    relation_texts = ids_to_texts(relation_ids, relation_texts)
    
    query_embeddings = encode_text_pairs(
        left_texts=head_texts,
        right_texts=relation_texts,
        encoder=hr_encoder,
        tokenizer=tokenizer,
        max_length=64,
        batch_size=14
    )
    
    RETURN query_embeddings    # [batch_size, 768], normalized

END FUNCTION


# ============================================================================
# FUNCTION: Encode Tail (entity only)
# ============================================================================

FUNCTION encode_tail(tail_ids, t_encoder, tokenizer, entity_texts):
    """
    Encode tail entity descriptions
    """
    
    tail_texts = ids_to_texts(tail_ids, entity_texts)
    
    all_embeddings = []
    
    FOR batch_start FROM 0 TO len(tail_texts) STEP batch_size:
        batch_end = MIN(batch_start + batch_size, len(tail_texts))
        batch_texts = tail_texts[batch_start:batch_end]
        
        # Tokenize single text inputs
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass
        hidden = t_encoder(**encoded).last_hidden_state
        
        # Mean pooling
        mask = encoded['attention_mask'].unsqueeze(-1)
        sum_embeddings = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_embeddings / denom
        
        # L2 normalize
        normalized = pooled / (pooled.norm(p=2, dim=-1, keepdim=True) + epsilon)
        
        all_embeddings.append(normalized)
    
    RETURN torch.cat(all_embeddings, dim=0)    # [batch_size, 768], normalized

END FUNCTION


# ============================================================================
# FUNCTION: Alignment Loss
# ============================================================================

FUNCTION alignment_loss(query_embeddings, tail_embeddings):
    """
    L_align = mean(||q - t||_2^2)
    Both q and t are L2-normalized vectors
    """
    
    diff = query_embeddings - tail_embeddings        # [batch_size, 768]
    distances = torch.norm(diff, p=2, dim=-1)        # [batch_size]
    squared_distances = distances ** 2               # [batch_size]
    loss = torch.mean(squared_distances)             # scalar
    
    RETURN loss

END FUNCTION


# ============================================================================
# FUNCTION: Uniformity Loss
# ============================================================================

FUNCTION uniformity_loss(embeddings, max_samples=170, chunk_size=128):
    """
    L_uni = log(mean(exp(-2 * ||x_i - x_j||_2^2)))
    Encourages uniform spread on hypersphere
    """
    
    n = embeddings.shape[0]
    
    IF n < 2:
        RETURN torch.tensor(0.0, device=embeddings.device)
    
    # Optional subsampling
    IF max_samples > 0 AND n > max_samples:
        indices = torch.randperm(n)[:max_samples]
        embeddings = embeddings[indices]
        n = max_samples
    
    pair_sum = 0.0
    pair_count = 0
    
    # Chunked computation to avoid O(n²) memory
    FOR i_start FROM 0 TO n STEP chunk_size:
        i_end = MIN(i_start + chunk_size, n)
        xi = embeddings[i_start:i_end]              # [chunk_size, 768]
        
        FOR j_start FROM i_start TO n STEP chunk_size:
            j_end = MIN(j_start + chunk_size, n)
            xj = embeddings[j_start:j_end]          # [chunk_size, 768]
            
            # Pairwise squared Euclidean distances
            diff = xi.unsqueeze(1) - xj.unsqueeze(0)    # [ci, cj, 768]
            dist_sq = (diff ** 2).sum(dim=-1)           # [ci, cj]
            
            # Exponential kernel
            weights = torch.exp(-2 * dist_sq)           # [ci, cj]
            
            IF i_start == j_start:
                # Skip diagonal (same embeddings)
                diagonal_mask = torch.eye(i_end - i_start, dtype=torch.bool)
                valid_weights = weights.masked_select(~diagonal_mask)
                pair_sum += valid_weights.sum() * 0.5
                pair_count += valid_weights.numel() // 2
            ELSE:
                pair_sum += weights.sum()
                pair_count += weights.numel()
    
    IF pair_count == 0:
        RETURN torch.tensor(0.0, device=embeddings.device)
    
    loss = torch.log(pair_sum / pair_count)
    RETURN loss

END FUNCTION


# ============================================================================
# FUNCTION: Extract Unique Embeddings from Batch
# ============================================================================

FUNCTION extract_unique_embeddings(head_ids, relation_ids, tail_ids, 
                                   encode_query_func, encode_tail_func):
    """
    Optimize by encoding unique queries and tails only once per batch
    """
    
    # Stack head + relation pairs
    query_pairs = torch.stack([head_ids, relation_ids], dim=1)
    
    // Find unique queries and mapping indices
    unique_queries, query_inverse = torch.unique(
        query_pairs,
        dim=0,
        return_inverse=True
    )
    
    // Encode unique queries
    unique_query_emb = encode_query_func(
        unique_queries[:, 0],
        unique_queries[:, 1]
    )  // [num_unique_queries, 768]
    
    // Find unique tails and mapping indices
    unique_tails, tail_inverse = torch.unique(
        tail_ids,
        return_inverse=True
    )
    
    // Encode unique tails
    unique_tail_emb = encode_tail_func(unique_tails)  // [num_unique_tails, 768]
    
    RETURN {
        'unique_query_emb': unique_query_emb,
        'unique_tail_emb': unique_tail_emb,
        'query_inverse': query_inverse,
        'tail_inverse': tail_inverse,
    }

END FUNCTION


# ============================================================================
# FUNCTION: Forward Pass (Compute Distances)
# ============================================================================

FUNCTION forward_pass(head_ids, relation_ids, tail_ids, 
                     unique_query_emb, unique_tail_emb,
                     query_inverse, tail_inverse,
                     forward_chunk_size=32768):
    """
    Compute L2 distances between queries and tails
    Chunked version to avoid large tensor materialization
    """
    
    n_batch = head_ids.shape[0]
    embedding_dim = unique_query_emb.shape[1]
    
    IF n_batch * embedding_dim <= forward_chunk_size:
        // No chunking needed
        query_emb = unique_query_emb[query_inverse]
        tail_emb = unique_tail_emb[tail_inverse]
        distances = torch.norm(query_emb - tail_emb, p=2, dim=-1)
        RETURN distances
    ELSE:
        // Chunked forward pass
        chunk_size = forward_chunk_size // embedding_dim
        distances = []
        
        FOR start FROM 0 TO n_batch STEP chunk_size:
            end = MIN(start + chunk_size, n_batch)
            q_idx = query_inverse[start:end]
            t_idx = tail_inverse[start:end]
            
            q_part = unique_query_emb[q_idx]
            t_part = unique_tail_emb[t_idx]
            
            dist_part = torch.norm(q_part - t_part, p=2, dim=-1)
            distances.append(dist_part)
        
        RETURN torch.cat(distances, dim=0)
    
END FUNCTION


# ============================================================================
# FUNCTION: Main Training Step
# ============================================================================

FUNCTION train_one_epoch(train_triples, model, optimizer, scaler, config):
    """
    Train for one epoch with gradient accumulation and AMP
    """
    
    head_ids, relation_ids, tail_ids = train_triples
    n_train = len(head_ids)
    
    // Shuffle
    perm = torch.randperm(n_train)
    head_ids = head_ids[perm].to(device)
    relation_ids = relation_ids[perm].to(device)
    tail_ids = tail_ids[perm].to(device)
    
    optimizer.zero_grad()
    epoch_loss = 0.0
    batch_count = 0
    
    FOR batch_idx, (h_batch, r_batch, t_batch) IN enumerate_batches(config.n_batch, head_ids, relation_ids, tail_ids):
        
        WITH torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.amp):
            
            // 1. ENCODE QUERY AND TAIL
            q_batch = encode_query(h_batch, r_batch, model.hr_encoder, ...)
            t_batch = encode_tail(t_batch, model.t_encoder, ...)
            
            // 2. ALIGNMENT LOSS
            loss_align = alignment_loss(q_batch, t_batch)
            
            // 3. EXTRACT UNIQUE EMBEDDINGS FOR UNIFORMITY
            unique_info = extract_unique_embeddings(
                h_batch, r_batch, t_batch,
                encode_query, encode_tail
            )
            
            // 4. UNIFORMITY LOSS
            loss_uni_q = uniformity_loss(unique_info['unique_query_emb'])
            loss_uni_t = uniformity_loss(unique_info['unique_tail_emb'])
            loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
            
            // 5. TOTAL LOSS
            gamma = config.gamma                    // typically 1.0
            loss = loss_align + gamma * loss_uni
        
        // 6. BACKWARD WITH GRADIENT ACCUMULATION
        scaled_loss = loss / config.grad_accum_steps
        scaler.scale(scaled_loss).backward()
        
        // 7. OPTIMIZER STEP (every grad_accum_steps)
        IF (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * h_batch.shape[0]
        batch_count += 1
        
        LOG: "Epoch {epoch}, Batch {batch_idx}, Loss = {loss.item():.6f}"
    
    // 8. FLUSH REMAINING GRADIENTS
    IF batch_count % config.grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
    
    avg_loss = epoch_loss / n_train
    RETURN avg_loss

END FUNCTION


# ============================================================================
# FUNCTION: Complete Training Loop
# ============================================================================

FUNCTION train_model(train_triples, valid_triples, test_triples, config):
    """
    Complete training with validation and early stopping
    """
    
    model = initialize_model()
    best_valid_perf = 0.0
    best_epoch = -1
    patience_counter = 0
    
    FOR epoch FROM 1 TO config.n_epoch:
        
        // Train one epoch
        avg_loss = train_one_epoch(train_triples, model, config)
        LOG: "Epoch {epoch}/{config.n_epoch}, Loss = {avg_loss:.6f}"
        
        // Validation
        valid_perf = validate(model, valid_triples, config)
        LOG: "Epoch {epoch}, Valid MRR = {valid_perf:.4f}"
        
        // Early stopping
        IF valid_perf > best_valid_perf:
            best_valid_perf = valid_perf
            best_epoch = epoch
            patience_counter = 0
            model.save_checkpoint()
        ELSE:
            patience_counter += 1
        
        // Test evaluation
        IF config.epoch_per_test > 0 AND epoch % config.epoch_per_test == 0:
            test_perf = test(model, test_triples, config)
            LOG: "Epoch {epoch}, Test MRR = {test_perf:.4f}"
        
        // Early stopping check
        IF config.early_stop_patience > 0 AND patience_counter >= config.early_stop_patience:
            LOG: "Early stopping at epoch {epoch}"
            BREAK
    
    // Load best model
    model.load_checkpoint()
    RETURN model

END FUNCTION
```

---

## Validation

```python
# ============================================================================
# FUNCTION: Validation
# ============================================================================

FUNCTION validate(model, valid_triples, config):
    """
    Evaluate on validation set (typically 1K-10K triples)
    Uses same metrics as test: MRR, MR, Hits@K
    """
    
    model.set_eval_mode()
    
    WITH torch.no_grad():
        metrics = evaluate_link_prediction(
            model=model,
            test_triples=valid_triples,
            config=config,
            filter_false_negatives=FALSE
        )
    
    valid_mrr = metrics['mrr']
    LOG: "Valid MRR = {valid_mrr:.4f}"
    
    RETURN valid_mrr

END FUNCTION
```

---

## Testing

```python
# ============================================================================
# FUNCTION: Link Prediction Testing (Main Evaluation)
# ============================================================================

FUNCTION test_link_prediction(model, test_triples, config):
    """
    Evaluate on all test triples
    Compute MR, MRR, Hits@K for both tail and head prediction
    """
    
    model.set_eval_mode()
    head_ids, relation_ids, tail_ids = test_triples
    n_test = len(head_ids)
    
    WITH torch.no_grad():
        
        // PRE-COMPUTE: Encode all entities once
        all_entity_ids = torch.arange(model.n_entity, device=device)
        entity_embeddings = encode_tail(
            all_entity_ids,
            model.t_encoder,
            model.tokenizer,
            model.entity_texts
        )  // [n_entity, 768]
        
        // Initialize metrics
        mr_total = 0.0
        mrr_total = 0.0
        hits_total = [0, 0, 0]  // Hits@[1, 3, 10]
        k_list = [1, 3, 10]
        
        // Process in batches
        test_batch_size = config.test_batch_size
        
        FOR batch_idx, (h_batch, r_batch, t_batch) IN enumerate_batches(test_batch_size, head_ids, relation_ids, tail_ids):
            
            h_batch = h_batch.to(device)
            r_batch = r_batch.to(device)
            t_batch = t_batch.to(device)
            
            // 1. TAIL PREDICTION (h, r, ?)
            // Query: (head, relation)
            q_tail = encode_query(h_batch, r_batch, model.hr_encoder, ...)
            
            // Scores: -dot(query, entity_embeddings^T)
            // Using negative dot product for ranking (lower = better)
            tail_scores = -(q_tail @ entity_embeddings.T)  // [batch_size, n_entity]
            
            // 2. HEAD PREDICTION (?, r, t)
            // Query: (tail, inverse_relation)
            inv_relation_texts = ["inverse " + text for text in model.relation_texts[r_batch]]
            q_head = encode_query_with_relation_texts(
                tail_ids=t_batch,
                relation_texts=inv_relation_texts,
                encoder=model.hr_encoder
            )
            
            head_scores = -(q_head @ entity_embeddings.T)  // [batch_size, n_entity]
            
            // 3. PROCESS EACH TRIPLE
            FOR i IN range(h_batch.shape[0]):
                
                head_id = h_batch[i].item()
                relation_id = r_batch[i].item()
                tail_id = t_batch[i].item()
                
                tail_scores_i = tail_scores[i].clone()
                head_scores_i = head_scores[i].clone()
                
                // 4. APPLY FILTERING (optional)
                IF config.filter_false_negatives:
                    
                    // For tail prediction: (h, r, ?)
                    // Get all valid tails for (h, r) from training data
                    valid_tail_key = (head_id, relation_id)
                    IF valid_tail_key IN model.valid_tails_sparse:
                        target_tail_score = tail_scores_i[tail_id].item()
                        
                        // Apply penalty to false negatives
                        penalty_mask = model.valid_tails_sparse[valid_tail_key]
                        tail_scores_i += penalty_mask * PENALTY  // 1e30
                        
                        // Restore original score for target
                        tail_scores_i[tail_id] = target_tail_score
                    
                    // For head prediction: (?, r, t)
                    // Get all valid heads for (t, r) from training data
                    valid_head_key = (tail_id, relation_id)
                    IF valid_head_key IN model.valid_heads_sparse:
                        target_head_score = head_scores_i[head_id].item()
                        
                        penalty_mask = model.valid_heads_sparse[valid_head_key]
                        head_scores_i += penalty_mask * PENALTY
                        
                        head_scores_i[head_id] = target_head_score
                
                // 5. COMPUTE RANKING METRICS
                tail_metrics = ranking_metrics(tail_scores_i, tail_id, k_list=[1, 3, 10])
                head_metrics = ranking_metrics(head_scores_i, head_id, k_list=[1, 3, 10])
                
                // Aggregate
                mr_total += tail_metrics['mr'] + head_metrics['mr']
                mrr_total += tail_metrics['mrr'] + head_metrics['mrr']
                
                FOR j IN range(len(k_list)):
                    hits_total[j] += tail_metrics['hits'][j] + head_metrics['hits'][j]
    
    // Normalize by total queries (2x test triples: head + tail prediction)
    total_queries = n_test * 2
    
    results = {
        'mr': mr_total / total_queries,
        'mrr': mrr_total / total_queries,
        'hits@1': hits_total[0] / total_queries,
        'hits@3': hits_total[1] / total_queries,
        'hits@10': hits_total[2] / total_queries,
    }
    
    LOG: "Test Results:"
    LOG: "  MR = {results['mr']:.4f}"
    LOG: "  MRR = {results['mrr']:.4f}"
    LOG: "  Hits@1 = {results['hits@1']:.4f}"
    LOG: "  Hits@3 = {results['hits@3']:.4f}"
    LOG: "  Hits@10 = {results['hits@10']:.4f}"
    
    RETURN results

END FUNCTION
```

---

## Loss Functions

```python
# ============================================================================
# QUICK REFERENCE: Loss Formulas
# ============================================================================

LOSS FUNCTION alignment_loss:
    INPUT: query_embeddings [B × 768], tail_embeddings [B × 768]
    OUTPUT: scalar loss
    
    Formula: L_align = mean(||q - t||₂²)
    
    where:
      • q and t are L2-normalized
      • ||·||₂ is Euclidean (L2) norm
      • mean is over batch dimension
    
    Code:
        diff = query_embeddings - tail_embeddings
        norms = torch.norm(diff, p=2, dim=-1)
        loss = torch.mean(norms ** 2)

END FUNCTION


LOSS FUNCTION uniformity_loss:
    INPUT: embeddings [N × 768]
    OUTPUT: scalar loss
    
    Formula: L_uni = log(mean_{i<j}[exp(-2 * ||x_i - x_j||₂²)])
    
    Interpretation:
      • Computes pairwise distances between all embeddings
      • Uses exponential kernel: far apart = low weight, close = high weight
      • Taking log of mean encourages spreading (uniformity)
      • Loss is higher when embeddings are spread uniformly
    
    Properties:
      • Can be computed in chunks to avoid O(N²) memory
      • Often subsampled for large N
      • Applied separately to queries and entities

END FUNCTION


LOSS FUNCTION total_loss:
    INPUT: alignment_loss, uniformity_loss, gamma
    OUTPUT: scalar total loss
    
    Formula: L_total = L_align + gamma * L_uni
    
    where:
      • gamma = uniformity weight (typically 1.0)
      • Controls balance between alignment and uniformity
      • gamma=0 → only alignment (triple matching)
      • gamma→∞ → only uniformity (collapse prevention)
    
    Typical behavior:
      • Alignment loss ≈ 0.1 - 1.0 (L2 distance squared)
      • Uniformity loss ≈ -1.0 to -5.0 (log of sum)
      • With gamma=1.0, both terms contribute

END FUNCTION
```

---

## Key Concepts Summary

| Concept | Explanation |
|---------|------------|
| **Query** | Embedding of (head_entity, relation) text pair |
| **Tail** | Embedding of tail_entity text |
| **L2 Normalization** | Divide vector by its norm to project onto unit sphere |
| **Alignment Loss** | Minimizes distance between query and correct tail |
| **Uniformity Loss** | Maximizes spread of embeddings to prevent collapse |
| **Gradient Accumulation** | Average gradients over multiple mini-batches before step |
| **AMP** | Automatic Mixed Precision: use FP16 where possible, FP32 where needed |
| **Gradient Checkpointing** | Recompute activations during backward to save memory |
| **Inverse Relation** | Used for head prediction: (t, "inverse r", h) |
| **Filtering** | Remove training examples from candidate set during ranking |
| **MR** | Mean Rank (lower is better) |
| **MRR** | Mean Reciprocal Rank (higher is better) |
| **Hits@K** | % of triples ranked in top-K (higher is better) |
