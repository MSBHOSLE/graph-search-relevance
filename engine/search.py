import math
from typing import Dict, List, Set, Tuple


class HybridGraphSearch:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # Weight for graph topology vs text similarity
        self.graph_adj: Dict[str, Set[str]] = {}
        self.doc_texts: Dict[str, str] = {}

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.graph_adj: self.graph_adj[u] = set()
        if v not in self.graph_adj: self.graph_adj[v] = set()
        self.graph_adj[u].add(v)
        self.graph_adj[v].add(u)

    def set_document(self, node: str, text: str) -> None:
        self.doc_texts[node] = text

    def _graph_jaccard(self, u: str, v: str) -> float:
        neighbors_u = self.graph_adj.get(u, set())
        neighbors_v = self.graph_adj.get(v, set())
        union = len(neighbors_u.union(neighbors_v))
        if union == 0:
            return 0.0
        return len(neighbors_u.intersection(neighbors_v)) / float(union)

    def _text_cosine_sim(self, text_a: str, text_b: str) -> float:
        tokens_a = text_a.lower().split()
        tokens_b = text_b.lower().split()
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = len(set_a.intersection(set_b))
        denom = math.sqrt(len(set_a)) * math.sqrt(len(set_b))
        return (intersection / denom) if denom > 0 else 0.0

    def rank(self, seed_node: str, query: str, threshold: float = 0.20) -> List[Tuple[str, float]]:
        """Ranks graph candidate nodes using hybrid graph topology + text relevance."""
        results = []
        for node, text in self.doc_texts.items():
            if node == seed_node:
                continue
            graph_sim = self._graph_jaccard(seed_node, node)
            text_sim = self._text_cosine_sim(query, text)
            combined = (self.alpha * graph_sim) + ((1.0 - self.alpha) * text_sim)

            # SVM / Threshold noise filter
            if combined >= threshold:
                results.append((node, round(combined, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
