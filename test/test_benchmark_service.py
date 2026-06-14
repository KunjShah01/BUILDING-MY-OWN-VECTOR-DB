"""Tests for the benchmark service."""
from __future__ import annotations
from services.benchmark_service import BenchmarkSuite
import numpy as np


class TestBenchmarkSuite:
    def setup_method(self):
        self.bm = BenchmarkSuite()

    def test_generate_synthetic_dataset(self):
        vectors, queries, gt = self.bm.generate_synthetic_dataset(100, 10, 32)
        assert vectors.shape == (100, 32)
        assert queries.shape == (10, 32)
        assert gt.shape == (10, 10)

    def test_run_benchmark(self):
        def search_fn(q, k=10, method="hnsw"):
            return {"results": [{"vector_id": str(i), "distance": 0.1} for i in range(k)]}
        vectors, queries, gt = self.bm.generate_synthetic_dataset(50, 5, 16)
        results = self.bm.run_recall_benchmark(search_fn, queries, gt, k=5, methods=["hnsw", "brute"])
        assert len(results) == 2
        assert all(r["queries_run"] > 0 for r in results)

    def test_get_results(self):
        results = self.bm.get_results()
        assert "benchmarks" in results
