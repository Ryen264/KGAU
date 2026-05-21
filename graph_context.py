from collections import defaultdict
from typing import Dict, List, Set


class LinkGraph:
    """Undirected 1-hop adjacency graph built from positive train triples."""

    def __init__(self, train_h: List[int], train_t: List[int]):
        self.graph: Dict[int, Set[int]] = defaultdict(set)
        for h, t in zip(train_h, train_t):
            if h == t:
                continue
            self.graph[h].add(t)
            self.graph[t].add(h)

    def get_neighbor_ids(self, entity_id: int, max_to_keep: int = 10) -> List[int]:
        if entity_id not in self.graph:
            return []
        if max_to_keep <= 0:
            return []
        return sorted(self.graph[entity_id])[:max_to_keep]

    def num_nodes(self) -> int:
        return len(self.graph)

    def num_edges_undirected(self) -> int:
        # Each undirected edge is stored twice in adjacency sets.
        return sum(len(v) for v in self.graph.values()) // 2
