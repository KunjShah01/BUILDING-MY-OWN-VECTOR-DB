"""Retrieval benchmark for the zero-dep Agentic Memory engine.

Measures *retrieval* quality (Recall@k, MRR) and latency (P50/P99) of the
pure-Python CharNGram embedder + MemoryStore on a synthetic memory corpus
with distractors. This is a fast, dependency-free smoke benchmark — NOT the
full LongMemEval-S run (see `longmemeval` subcommand / roadmap Q4 2026).

Metric note: numbers here are *retrieval recall*, the same family of metric
agentmemory reports (R@k). They are NOT directly comparable to Mem0's
*end-to-end QA accuracy* (an LLM judges the final answer). Keep the metric
labels honest when comparing — see PUBLISHED_BASELINES below.
"""

import sys
import time

# Windows terminals default to cp1252 and choke on unicode. Force utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - older interpreters / odd streams
    pass

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mem.embedder import CharNGramEmbedder, SemanticEmbedder
from mem.store import MemoryStore, MemoryRecord

TEST_USER_ID = 9999
LATENCY_TRIALS = 200  # repeat each search to get stable percentiles

# Gold memories the queries should retrieve, plus distractors that share
# vocabulary so retrieval is non-trivial (lexical overlap alone won't win).
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

# Paraphrase-heavy queries with minimal lexical overlap with the gold text.
# This is where a lexical char-ngram embedder struggles and a semantic model
# wins. Categories intentionally omitted so retrieval must rank the whole pool.
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

# Real, sourced published numbers. Labelled by metric so the comparison is
# honest. Retrieval recall (R@k) and QA accuracy are different metrics.
PUBLISHED_BASELINES = """Published baselines (cite the metric, not just the number):
  agentmemory  R@5  95.2%  retrieval recall on LongMemEval-S (BM25+vector, all-MiniLM-L6-v2)
               R@5  86.2%  retrieval recall, BM25 keyword-only baseline
               R@10 98.6%  retrieval recall, hybrid
               src: github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md
  Mem0         94.4%        END-TO-END QA accuracy on LongMemEval (GPT-4o judge), ~73% token savings
               92.5%        END-TO-END QA accuracy on LoCoMo
               src: mem0.ai/research
  NOTE: both backends below are retrieval layers (own engine, no memory SDK),
  comparable to the R@k rows above, NOT to Mem0's QA accuracy. char-ngram is
  zero-dep/lexical and weak on paraphrase; semantic uses all-MiniLM-L6-v2 and
  closes most of that gap on the same model family agentmemory reports."""


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _eval_backend(embedder, show_rows=False):
    """Seed a fresh store with this embedder, run all queries, return metrics."""
    store = MemoryStore(persist_path=None)  # in-memory only
    store.initialize()
    store.delete_user(TEST_USER_ID)
    for m in GOLD_MEMORIES + DISTRACTOR_MEMORIES:
        store.insert([MemoryRecord(
            point_id=m["id"], user_id=TEST_USER_ID,
            memory_text=m["text"], categories=m["categories"],
            date="2026-01-01", embedding=embedder.embed(m["text"]),
        )])

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    all_latencies = []

    if show_rows:
        print(f"\n{'Query':<52} {'Gold':<6} {'Rank':<6} {'R@5':<5}")
        print("-" * 74)

    for tq in TEST_QUERIES:
        query_vec = embedder.embed(tq["query"])
        gold_ids = set(tq["gold"])

        for _ in range(LATENCY_TRIALS):
            start = time.perf_counter()
            results = store.search(query_vec, user_id=TEST_USER_ID, limit=10)
            all_latencies.append((time.perf_counter() - start) * 1000)

        rank = next((i for i, r in enumerate(results, 1) if r.point_id in gold_ids), None)
        for k in (1, 3, 5):
            if rank is not None and rank <= k:
                hits_at[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        if show_rows:
            rank_str = str(rank) if rank else "miss"
            at5 = "OK" if (rank and rank <= 5) else "X"
            print(f"{tq['query']:<52} {tq['gold'][0]:<6} {rank_str:<6} {at5:<5}")

    n = len(TEST_QUERIES)
    all_latencies.sort()
    return {
        "backend": embedder.name,
        "recall_at_1": hits_at[1] / n,
        "recall_at_3": hits_at[3] / n,
        "recall_at_5": hits_at[5] / n,
        "mrr": sum(reciprocal_ranks) / n,
        "p50_ms": _percentile(all_latencies, 50),
        "p99_ms": _percentile(all_latencies, 99),
    }


def run_benchmark():
    n_gold, n_dist = len(GOLD_MEMORIES), len(DISTRACTOR_MEMORIES)
    print(f"Seeded {n_gold} gold + {n_dist} distractor memories ({n_gold + n_dist} total).")
    print("Queries are paraphrase-heavy (low lexical overlap with gold text).")

    results = []

    # Lexical backend (always available, zero-dep).
    results.append(_eval_backend(CharNGramEmbedder(), show_rows=True))

    # Semantic backend (sentence-transformers); skip cleanly if unavailable.
    try:
        sem = SemanticEmbedder()
        sem._ensure_model()
        results.append(_eval_backend(sem))
    except Exception as e:
        print(f"\n[semantic backend skipped: {type(e).__name__} — model unavailable]")

    print(f"\n{'Backend':<12} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'P50(ms)':>9} {'P99(ms)':>9}")
    print("-" * 64)
    for r in results:
        print(f"{r['backend']:<12} {r['recall_at_1']*100:>6.1f}% {r['recall_at_3']*100:>6.1f}% "
              f"{r['recall_at_5']*100:>6.1f}% {r['mrr']:>7.3f} {r['p50_ms']:>9.3f} {r['p99_ms']:>9.3f}")

    print()
    print(PUBLISHED_BASELINES)
    return results


if __name__ == "__main__":
    run_benchmark()
