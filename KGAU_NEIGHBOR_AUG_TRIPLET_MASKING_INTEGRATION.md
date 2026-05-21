# KGAU Integration Guide: Neighbor Augmentation + Triplet Masking

## Goal

Add two SimKGC-inspired capabilities to KGAU DirectAU:

1. Neighbor augmentation: enrich entity text with 1-hop graph neighbors.
2. Triplet masking: prevent target leakage when constructing augmented text for a training triple.

This guide is tailored to KGAU's current pipeline (JSON preprocessed examples, `DirectAUKG` model, no in-batch contrastive logits).

## Current KGAU Baseline (Important for Design)

KGAU currently:

- Loads preprocessed examples from `train.txt.json`, `valid_w_label.txt.json`, `test_w_label.txt.json`.
- Loads entity metadata from `entities.json` and relation strings from `relations.json`.
- Builds:
  - `entity_texts = [entity_desc or entity]`
  - `relation_texts = list(relation_map.values())`
- Trains DirectAU with:
  - Alignment loss: `mean(||q - t||^2)`
  - Uniformity loss on unique queries/entities
- Uses filtered ranking at eval (`sparse_heads_tails`) but does not use InfoNCE-style in-batch negative logits.

Implication:

- Classic triplet-mask-on-logits is not the primary integration point in DirectAU training.
- The practical leakage risk here is in text augmentation (head text accidentally includes true tail, or tail text includes true head).

## Design Summary for KGAU

Implement two masking scopes:

1. Augmentation-time triplet masking (must-have for DirectAU)
- For each training triple `(h, r, t)`:
  - Query side: augment `h` with neighbors excluding `t`
  - Tail side: augment `t` with neighbors excluding `h`

2. Candidate-level triplet masking (already present at eval)
- Keep filtered ranking behavior unchanged (`heads`/`tails` sparse masks in eval).

Optional future scope:
- If KGAU adds InfoNCE logits later, add matrix-level triplet mask there.

## Implementation Plan

## Step 1: Add Config Flags

Add these fields under `DirectAU_KG` config:

```yaml
use_link_graph: true
link_graph_max_neighbors: 10
neighbor_min_tokens: 20
neighbor_text_field: entity
triplet_masking_for_neighbors: true
```

Flag semantics:

- `use_link_graph`: enable neighbor augmentation path.
- `link_graph_max_neighbors`: cap neighbors per entity.
- `neighbor_min_tokens`: only augment short descriptions.
- `neighbor_text_field`: use `entity` names or `entity_desc` snippets for neighbor text.
- `triplet_masking_for_neighbors`: exclude opposite endpoint to prevent leakage.

## Step 2: Build a Neighbor Graph from Train Positives

Add a utility module (suggested: `graph_context.py`) with:

- `LinkGraph` built from training positive triples only.
- ID-level adjacency: `entity_idx -> set(neighbor_entity_idx)`.
- Deterministic neighbor order (sorted IDs) for reproducibility.

Example API:

```python
class LinkGraph:
    def __init__(self, train_h, train_t):
        self.graph = defaultdict(set)
        for h, t in zip(train_h, train_t):
            self.graph[h].add(t)
            self.graph[t].add(h)

    def get_neighbor_ids(self, entity_id: int, max_to_keep: int = 10) -> List[int]:
        if entity_id not in self.graph:
            return []
        return sorted(self.graph[entity_id])[:max_to_keep]
```

Where to build it:

- In `main.py`, after `train_triplets` is encoded and before model construction.
- Pass it into `DirectAUKG` constructor.

## Step 3: Add Context Builder in Model Module

In `DirectAU_KGModule`, add neighbor-aware text helpers.

State needed:

- `base_entity_texts`: original texts from `entities.json`.
- `link_graph`: adjacency object.
- config flags from Step 1.

Helper behavior:

```python
def _build_aug_text(self, entity_id: int, exclude_id: Optional[int] = None) -> str:
    base = self.base_entity_texts[entity_id]

    if not self.use_link_graph:
        return base

    if len(base.split()) >= self.neighbor_min_tokens:
        return base

    nbr_ids = self.link_graph.get_neighbor_ids(entity_id, self.link_graph_max_neighbors)
    if self.triplet_masking_for_neighbors and exclude_id is not None:
        nbr_ids = [nid for nid in nbr_ids if nid != exclude_id]

    if not nbr_ids:
        return base

    nbr_texts = [self._neighbor_surface_text(nid) for nid in nbr_ids]
    return (base + " " + " ".join(nbr_texts)).strip()
```

Important:

- Use this dynamic builder in training-time encoding paths.
- Keep deterministic output.

## Step 4: Wire Dynamic Masked Augmentation into Encode Paths

Current signatures:

- `encode_query(head, relation)`
- `encode_tail(tail)`

Recommended signature changes:

- `encode_query(head, relation, exclude_tail=None)`
- `encode_tail(tail, exclude_head=None)`

Training call-site update in `DirectAUKG.train`:

```python
q_full = self.model.encode_query(h_batch, r_batch, exclude_tail=t_batch)
t_full = self.model.encode_tail(t_batch, exclude_head=h_batch)
```

Inference call-sites:

- Keep exclusion `None`.
- Use full augmented text (no per-candidate exclusion needed).

This gives leakage-safe training while preserving efficient inference.

## Step 5: Keep Evaluation Filtering, Do Not Remove It

KGAU already applies filtered ranking with sparse masks (`heads`, `tails`) during `test_link`.

Keep this logic intact because it addresses ranking contamination from known positives, which is complementary to text-level masking.

## Step 6: Optional Cache for Throughput

Dynamic per-batch text generation can add CPU overhead.

Add a bounded cache keyed by tuple:

- `(entity_id, exclude_id)` during training.
- `(entity_id, None)` during inference.

Use LRU cache or dict with periodic clear per epoch.

## Step 7: Logging and Debug Checks

Add startup logs:

- `use_link_graph`
- `link_graph_max_neighbors`
- `neighbor_min_tokens`
- `% entities augmented`
- average neighbors used

Add one safety assertion in debug mode:

- During training augmentation, ensure `exclude_id` text is not present in appended neighbor list by ID check before text join.

## Minimal Patch Map

1. `main.py`
- Parse/read new config flags.
- Build `LinkGraph` from `train_triplets`.
- Pass graph and flags into model constructor.

2. `model.py`
- Extend `DirectAU_KGModule.__init__` to store link graph + flags.
- Add `_build_aug_text` and optional cache.
- Extend `encode_query` and `encode_tail` with exclusion arguments.
- Update `DirectAUKG.train` to pass exclusions.

3. New utility file (recommended `graph_context.py`)
- `LinkGraph` implementation.

4. Config files under `config/`
- Add the new keys under `DirectAU_KG` (or `DirectAUKG` with compatibility mapping).

## KGAU-Specific Notes

1. Key naming mismatch exists in repository docs/configs:
- Some configs use `DirectAUKG`
- Model code indexes `DirectAU_KG`

Recommendation:
- Keep current compatibility bridge in `main.py` (`DirectAUKG` -> `DirectAU_KG`) and add new flags in both blocks when needed.

2. Current train pipeline reads JSON examples and then maps IDs.
- Build graph after ID mapping to avoid repeated string conversion.

3. DirectAU in KGAU has no negative-logit matrix.
- So "triplet masking" should primarily mean leakage-safe neighbor exclusion for training text.
- If an InfoNCE head is added later, introduce matrix triplet masks there.

## Validation Protocol

## Unit Checks

1. Graph correctness
- For each train edge `(h, t)`, verify mutual adjacency in `LinkGraph`.

2. Leakage-safe augmentation
- For sampled training triples, confirm excluded endpoint ID is absent from selected neighbors.

3. Determinism
- Same seed and config produce identical augmented text outputs.

## Functional Checks

1. Baseline parity
- With `use_link_graph=false`, metrics and speed should match current behavior.

2. Augmentation on
- Expect improved or stable MRR/Hit@K on sparse entities.

3. Overhead tracking
- Measure tokens-per-batch and training throughput before/after.

## Suggested Ablation Matrix

Run at least:

1. DirectAU baseline
- `use_link_graph=false`

2. Neighbor augmentation only
- `use_link_graph=true`, `triplet_masking_for_neighbors=false`

3. Neighbor augmentation + masking (target)
- `use_link_graph=true`, `triplet_masking_for_neighbors=true`

Recommended report columns:

- MR, MRR, Hit@1/3/10
- Classification ACC/F1/ROC_AUC
- Train time per epoch
- Peak GPU memory

## Recommended Default Values

Start with:

```yaml
use_link_graph: true
link_graph_max_neighbors: 10
neighbor_min_tokens: 20
neighbor_text_field: entity
triplet_masking_for_neighbors: true
```

Then tune:

- `link_graph_max_neighbors`: 5, 10, 20
- `neighbor_min_tokens`: 12, 20, 32

## Risks and Mitigations

1. Noise from weak neighbors
- Mitigation: small neighbor cap, relation-aware pruning later.

2. Longer sequences increase truncation
- Mitigation: only augment short base texts (`neighbor_min_tokens`).

3. CPU bottleneck from dynamic text generation
- Mitigation: cache augmented strings and keep encode micro-batch controlled.

## Migration Checklist

- [ ] Add config flags
- [ ] Implement `LinkGraph`
- [ ] Pass graph into `DirectAUKG`
- [ ] Add masked augmentation helpers in module
- [ ] Update train encode calls with exclusions
- [ ] Keep eval filtered ranking unchanged
- [ ] Add logging and safety checks
- [ ] Run ablations and record metrics

## Done Criteria

Integration is complete when:

1. KGAU trains with neighbor augmentation enabled and no runtime regressions.
2. Training-time augmentation excludes opposite endpoint IDs for each triple.
3. Baseline-off mode reproduces previous metrics.
4. Ablation confirms masking prevents leakage while preserving or improving ranking quality.
