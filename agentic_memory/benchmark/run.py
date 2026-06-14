"""Retrieval benchmark for the vector-DB-backed agentic memory engine.

Measures retrieval quality (Recall@k, MRR) and latency of the vector DB
memory service (pgvector + sentence-transformers) on a synthetic memory
corpus with distractors. Requires the vector DB to be running.
"""

from __future__ import annotations

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import httpx
except ImportError:
    httpx = None

BASE_URL = os.getenv("VECTOR_DB_URL", "http://localhost:8000")
TEST_USER = "benchmark_user"
LATENCY_TRIALS = 50

GOLD_MEMORIES = [
    {"id": "g1", "text": "User prefers aisle seats when flying", "categories": ["preferences", "travel"]},
    {"id": "g2", "text": "User works at a startup called Acme Corp", "categories": ["work"]},
    {"id": "g3", "text": "User has a dog named Max who is a golden retriever", "categories": ["personal", "pets"]},
    {"id": "g4", "text": "User is learning Japanese and has been studying for 6 months", "categories": ["learning", "hobbies"]},
    {"id": "g5", "text": "User's preferred coding languages are Python and TypeScript", "categories": ["work", "technology"]},
    {"id": "g6", "text": "User contributed to the open-source project Mem0 in 2025", "categories": ["work", "open-source"]},
    {"id": "g7", "text": "User lived in Tokyo for two years before moving to Ahmedabad", "categories": ["personal", "travel"]},
    {"id": "g8", "text": "User is allergic to shellfish", "categories": ["personal", "health"]},
    {"id": "g9", "text": "User's favorite book is 'The Pragmatic Programmer'", "categories": ["personal", "reading"]},
    {"id": "g10", "text": "User uses VS Code with Vim keybindings", "categories": ["work", "technology"]},
]

DISTRACTOR_MEMORIES = [
    {"id": "d1", "text": "User booked a window seat once but regretted it", "categories": ["travel"]},
    {"id": "d2", "text": "User interviewed at a big tech company called Globex", "categories": ["work"]},
    {"id": "d3", "text": "User's neighbor has a cat named Whiskers", "categories": ["personal", "pets"]},
    {"id": "d4", "text": "User studied Spanish in high school but forgot most of it", "categories": ["learning"]},
    {"id": "d5", "text": "User dislikes writing Java and avoids it", "categories": ["work", "technology"]},
    {"id": "d6", "text": "User read about the Qdrant vector database last year", "categories": ["open-source"]},
    {"id": "d7", "text": "User visited Kyoto on a short holiday", "categories": ["travel"]},
    {"id": "d8", "text": "User has no known food allergies to nuts", "categories": ["health"]},
    {"id": "d9", "text": "User started but never finished 'Clean Code'", "categories": ["reading"]},
    {"id": "d10", "text": "User tried Emacs years ago and switched away", "categories": ["technology"]},
]

TEST_QUERIES = [
    {"query": "Which spot on an airplane does the user like best?", "gold": ["g1"]},
    {"query": "Does the user own any animals?", "gold": ["g3"]},
    {"query": "Which coding tools does the user write software in?", "gold": ["g5"]},
    {"query": "In which cities has the user resided?", "gold": ["g7"]},
    {"query": "What food should the user avoid for safety?", "gold": ["g8"]},
    {"query": "Has the user given back to free software projects?", "gold": ["g6"]},
    {"query": "Which foreign tongue is the user picking up?", "gold": ["g4"]},
    {"query": "What novel does the user enjoy reading most?", "gold": ["g9"]},
    {"query": "Which text editor is the user's daily driver?", "gold": ["g10"]},
    {"query": "What company employs the user?", "gold": ["g2"]},
]


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _seed_memories(client: httpx.Client):
    for m in GOLD_MEMORIES + DISTRACTOR_MEMORIES:
        resp = client.post(
            f"{BASE_URL}/memories",
            json={"text": m["text"], "categories": m["categories"]},
            params={"user_id": TEST_USER},
        )
        resp.raise_for_status()


def _clear_memories(client: httpx.Client):
    resp = client.get(f"{BASE_URL}/memories", params={"user_id": TEST_USER, "limit": 200})
    if resp.status_code != 200:
        return
    data = resp.json()
    ids = [m["memory_id"] for m in data.get("memories", [])]
    if ids:
        client.post(f"{BASE_URL}/memories/batch-delete", json={"memory_ids": ids})


def run_benchmark():
    if httpx is None:
        print("httpx required: pip install httpx")
        return []

    client = httpx.Client(timeout=30)
    _clear_memories(client)
    _seed_memories(client)

    n_gold, n_dist = len(GOLD_MEMORIES), len(DISTRACTOR_MEMORIES)
    print(f"Seeded {n_gold} gold + {n_dist} distractor memories ({n_gold + n_dist} total).")

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    all_latencies = []

    print(f"\n{'Query':<52} {'Gold':<6} {'Rank':<6} {'R@5':<5}")
    print("-" * 74)

    for tq in TEST_QUERIES:
        gold_ids = set(tq["gold"])

        for _ in range(LATENCY_TRIALS):
            start = time.perf_counter()
            resp = client.post(
                f"{BASE_URL}/memories/search",
                json={"query": tq["query"], "limit": 10},
                params={"user_id": TEST_USER},
            )
            all_latencies.append((time.perf_counter() - start) * 1000)

        data = resp.json()
        results = data.get("results", [])

        rank = next(
            (i for i, r in enumerate(results, 1) if r["memory_id"] in gold_ids),
            None,
        )
        for k in (1, 3, 5):
            if rank is not None and rank <= k:
                hits_at[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        rank_str = str(rank) if rank else "miss"
        at5 = "OK" if (rank and rank <= 5) else "X"
        print(f"{tq['query']:<52} {tq['gold'][0]:<6} {rank_str:<6} {at5:<5}")

    n = len(TEST_QUERIES)
    all_latencies.sort()

    results = {
        "backend": "vector-db (all-MiniLM-L6-v2 + pgvector)",
        "recall_at_1": hits_at[1] / n,
        "recall_at_3": hits_at[3] / n,
        "recall_at_5": hits_at[5] / n,
        "mrr": sum(reciprocal_ranks) / n,
        "p50_ms": _percentile(all_latencies, 50),
        "p99_ms": _percentile(all_latencies, 99),
    }

    print(f"\n{'Backend':<32} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'P50(ms)':>9} {'P99(ms)':>9}")
    print("-" * 84)
    print(f"{results['backend']:<32} {results['recall_at_1']*100:>6.1f}% "
          f"{results['recall_at_3']*100:>6.1f}% {results['recall_at_5']*100:>6.1f}% "
          f"{results['mrr']:>7.3f} {results['p50_ms']:>9.3f} {results['p99_ms']:>9.3f}")

    return [results]


if __name__ == "__main__":
    run_benchmark()
