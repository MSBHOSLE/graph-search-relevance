import unittest
from engine.search import HybridGraphSearch


class TestHybridGraphSearch(unittest.TestCase):
    def setUp(self):
        self.engine = HybridGraphSearch(alpha=0.6)
        # Create a triangle graph: A-B, B-C, A-C
        self.engine.add_edge("NodeA", "NodeB")
        self.engine.add_edge("NodeB", "NodeC")
        self.engine.add_edge("NodeA", "NodeC")

        self.engine.set_document("NodeA", "deep learning neural network representation")
        self.engine.set_document("NodeB", "graph neural network node embeddings")
        self.engine.set_document("NodeC", "database query postgresql optimizer")

    def test_hybrid_ranking(self):
        results = self.engine.rank(seed_node="NodeA", query="graph neural network", threshold=0.1)
        self.assertGreaterEqual(len(results), 1)
        # NodeB shares text keywords + graph neighbor connectivity with NodeA
        top_node = results[0][0]
        self.assertEqual(top_node, "NodeB")


if __name__ == "__main__":
    unittest.main()
