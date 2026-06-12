# Vector Database Roadmap

This roadmap outlines the strategic direction for evolving this project from a robust, single-node vector database into a distributed, enterprise-ready data infrastructure system.

## Phase 1: Storage & Durability ✅ (Done)
To be truly production-grade, the database must survive crashes without data loss and manage datasets larger than available RAM.

- ✅ **Write-Ahead Logging (WAL)**: Append-only log with fsync durability, checkpoint/truncate after snapshot, and crash-recovery replay wired into HNSW index load (`utils/wal.py`, `database/hnsw_database.py`).
- ✅ **Memory-Mapped Vector Storage**: SSD-optimized `numpy.memmap`-backed store with dynamic growth, row reclamation, and persistence so only the working set stays resident (`utils/mmap_store.py`). _DiskANN/Vamana graph layout remains future work._
- ✅ **Background Compaction**: Soft-delete tombstones in HNSW preserve connectivity; a daemon thread hard-removes them by interval/ratio threshold (`utils/compaction.py`, `HNSWIndex.compact`).

## Phase 2: Distributed Systems & Scalability
A single node has physical limits. We need horizontal scalability.

- **Horizontal Sharding**: Implement consistent hashing to shard collections across multiple nodes.
- **Distributed Query Aggregation**: A coordinator node scatter-gathers queries to shards, fusing the top-K results globally.
- **Raft Consensus Engine**: Embed a Raft implementation (e.g., using `pysyncobj` or custom logic) for leader election and distributed WAL replication to ensure high availability.

## Phase 3: Advanced Ingestion & Integrations
Production environments demand seamless data pipelines, not just REST APIs.

- **Change Data Capture (CDC)**: Direct integration with Kafka and Debezium for real-time streaming ingestion from upstream operational databases.
- **Dynamic Quantization**: Auto-downgrade fp32 embeddings to Int8 or Binary vectors dynamically based on system memory pressure.
- **Agentic Connectors**: Out-of-the-box MCP (Model Context Protocol) and LangChain/LlamaIndex tools for instant agent integration.

## Phase 4: Query Planner & Optimization (In Progress)
Move from static search pipelines to intelligent, cost-based execution.

- ✅ **AST-Based Query Planner**: Recursive-descent parser turns hybrid queries like `(category = 'tech' AND price < 100) OR semantic_match("laptops")` into a typed AST (`utils/query_planner.py`).
- ✅ **Cost-Based Optimizer**: Selectivity heuristics (with optional per-field cardinality stats) pick `filter_first` vs `vector_first` vs `filter_only`/`vector_only`, exposed via `SearchEngineService.planned_search`.
- **Role-Based Access Control (RBAC)**: Fine-grained token permissions down to the row-level (document-level security).

## Phase 5: Hardware Acceleration
- **SIMD/AVX-512 Optimization**: Hand-optimized C++ bindings (via PyBind11 or Cython) for inner loop distance calculations.
- **GPU Indexing**: Expand CuPy usage to offload full index construction (like RAPIDS cuVS) rather than just batch distance calculations.

## Phase 6: Graph Neural Networks (GNN)
Move beyond simple graph traversal by applying ML directly over the HNSW index structure.

- **Graph Convolutional Networks (GCN)**: Train node embeddings that aggregate features from their HNSW neighbors, enabling deep contextual retrieval.
- **Temporal Graph Dynamics**: Track edge formations over time to identify trending clusters and bursty topics.
- **Link Prediction**: Automatically suggest missing metadata tags by inferring edges between disconnected but semantically similar sub-graphs.

## Phase 7: Production Search Engine Orchestration
Build a world-class, deployment-ready search orchestrator on top of the vector foundation.

- **Learning-to-Rank (LTR)**: Replace the static RRF scoring with an XGBoost/LightGBM model trained on implicit user feedback (click-through rates).
- **Personalization Engine**: Inject user profile embeddings into the query formulation phase to bias results towards user preferences.
- **Real-Time Data Connectors**: Built-in web crawlers and headless browser integrations for continuous ingestion of live web data.
