# Vector Database Roadmap

This roadmap tracks the evolution of this project from a robust, single-node vector database into a distributed, enterprise-ready data infrastructure system.

**Legend:** ✅ done · 🟡 partial · ⚪ planned

## Status at a Glance

| Phase | Theme | Status |
|-------|-------|--------|
| 1 | Storage & Durability | ✅ core done (DiskANN graph layout pending) |
| 2 | Distributed Systems & Scalability | ⚪ planned (partition primitives exist) |
| 3 | Advanced Ingestion & Integrations | 🟡 partial |
| 4 | Query Planner & Optimization | ✅ planner done (RBAC pending) |
| 5 | Hardware Acceleration | 🟡 partial (GPU batch distance) |
| 6 | Graph Neural Networks | 🟡 partial (graph rerank only) |
| 7 | Production Search Orchestration | ⚪ planned |
| 8 | Observability & Operations | ⚪ planned (new) |
| 9 | Security & Compliance | ⚪ planned (new) |

---

## Phase 1: Storage & Durability ✅ (Core Done)
Survive crashes without data loss and manage datasets larger than available RAM.

- ✅ **Write-Ahead Logging (WAL)**: Append-only log with fsync durability, checkpoint/truncate after snapshot, and crash-recovery replay wired into HNSW index load (`utils/wal.py`, `database/hnsw_database.py`).
- ✅ **Memory-Mapped Vector Storage**: `numpy.memmap`-backed store with dynamic growth, row reclamation, and persistence so only the working set stays resident (`utils/mmap_store.py`).
- ✅ **Background Compaction**: Soft-delete tombstones in HNSW preserve graph connectivity; a daemon thread hard-removes them by interval/ratio threshold (`utils/compaction.py`, `HNSWIndex.compact`).
- ⚪ **DiskANN / Vamana Graph Layout**: SSD-optimized on-disk graph (beam search over memory-mapped adjacency) for billion-scale serving. _Remaining work._

**Next up here:** wire WAL into the IVF database path; integrate `MmapVectorStore` as the backing store for HNSW vectors; build the Vamana on-disk graph.

## Phase 2: Distributed Systems & Scalability ⚪
A single node has physical limits. We need horizontal scalability.

- ⚪ **Horizontal Sharding**: Consistent hashing to shard collections across nodes. _Foundation: `utils/partitioned_index.py` already does hash/range partitioning in-process._
- ⚪ **Distributed Query Aggregation**: A coordinator node scatter-gathers queries to shards, fusing top-K results globally.
- ⚪ **Raft Consensus Engine**: Leader election + distributed WAL replication (e.g. `pysyncobj`) for high availability. _Builds directly on the Phase 1 WAL._

## Phase 3: Advanced Ingestion & Integrations 🟡
Production environments demand seamless data pipelines, not just REST APIs.

- ⚪ **Change Data Capture (CDC)**: Kafka/Debezium integration for real-time streaming ingestion from upstream databases.
- ⚪ **Dynamic Quantization**: Auto-downgrade fp32 → Int8/Binary based on memory pressure. _Foundation: `utils/int8_index.py` + PQ already exist; needs an adaptive policy layer._
- 🟡 **Agentic Connectors**: LangChain VectorStore done (`sdk/.../langchain_vectorstore.py`). LlamaIndex adapter and an MCP server remain.
- ✅ **Async Ingestion Queue**: In-memory batched queue with periodic flush + REST API (`services/ingestion_service.py`).

## Phase 4: Query Planner & Optimization ✅ (Planner Done)
Move from static search pipelines to intelligent, cost-based execution.

- ✅ **AST-Based Query Planner**: Recursive-descent parser turns hybrid queries like `(category = 'tech' AND price < 100) OR semantic_match("laptops")` into a typed AST (`utils/query_planner.py`).
- ✅ **Cost-Based Optimizer**: Selectivity heuristics (with optional per-field cardinality stats) pick `filter_first` / `vector_first` / `filter_only` / `vector_only`, exposed via `SearchEngineService.planned_search`.
- ⚪ **Role-Based Access Control (RBAC)**: Fine-grained token permissions down to row-level (document-level security). _Foundation: API keys already scope per tenant/collection; extend to row predicates._

**Next up here:** collect live per-field cardinality stats from PostgreSQL to feed the optimizer; expose a `/search/hybrid-query` REST endpoint accepting the query DSL.

## Phase 5: Hardware Acceleration 🟡
- 🟡 **GPU Indexing**: CuPy batch distance kernels exist (`utils/gpu_kernels.py`). Offload full index construction next (RAPIDS cuVS style).
- ⚪ **SIMD/AVX-512 Optimization**: Hand-optimized C++/Cython bindings (PyBind11) for inner-loop distance.

## Phase 6: Graph Neural Networks (GNN) 🟡
Apply ML directly over the HNSW index structure.

- 🟡 **Graph Rerank**: `services/gnn_service.py` re-ranks candidates over the graph. Full **GCN** node-embedding training remains.
- ⚪ **Temporal Graph Dynamics**: Track edge formations over time to surface trending/bursty clusters. _Foundation: time-series vectors already exist._
- ⚪ **Link Prediction**: Infer missing metadata tags by predicting edges between semantically similar sub-graphs.

## Phase 7: Production Search Orchestration ⚪
Build a deployment-ready search orchestrator on top of the vector foundation.

- ⚪ **Learning-to-Rank (LTR)**: Replace static RRF with XGBoost/LightGBM trained on click-through feedback. _Foundation: `/playground/feedback` already captures signals._
- ⚪ **Personalization Engine**: Inject user-profile embeddings into query formulation.
- ⚪ **Real-Time Data Connectors**: Built-in crawlers / headless-browser ingestion of live web data.

## Phase 8: Observability & Operations ⚪ (New)
Make the system debuggable and operable at scale.

- ⚪ **Distributed Tracing**: OpenTelemetry spans across API → service → index → DB (extends existing `utils/telemetry.py`).
- ⚪ **Index Health & Auto-Tuning**: Surface recall drift, tombstone ratio, and fragmentation; auto-trigger compaction/reindex (extends `services/auto_reindex.py`).
- ⚪ **Backup & Restore**: Point-in-time recovery from WAL + snapshot; scheduled S3/Azure backups of index files.
- ⚪ **Graceful Shutdown / Startup Recovery**: On boot, replay all collection WALs and rebuild in-memory indexes automatically.

## Phase 9: Security & Compliance ⚪ (New)
Enterprise readiness.

- ⚪ **Encryption at Rest**: Encrypt index files and mmap store on disk.
- ⚪ **Audit Logging**: Immutable log of every mutation and access for compliance.
- ⚪ **PII Redaction & Data Residency**: Per-tenant region pinning and field-level redaction in metadata.
- ⚪ **mTLS between nodes**: Secure inter-node traffic once sharding lands (Phase 2).

---

## Recommended Next Sprint

Highest-leverage work, ordered:

1. **Finish Phase 1 durability loop** — wire WAL into IVF DB + auto-replay all collection WALs on app startup (Phase 8 overlap). Small, high value: makes the durability story real end-to-end.
2. **Expose the query planner over REST** — `/search/hybrid-query` endpoint + live cardinality stats. Turns Phase 4 code into a user-facing feature.
3. **RBAC row-level permissions** (Phase 4) — extend existing API-key scoping; unblocks enterprise use.
4. **Distributed query aggregation** (Phase 2) — coordinator over existing `partitioned_index.py`; the biggest scalability unlock.
