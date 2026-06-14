# Changelog

## [Unreleased]

### Added
- **Agentic Memory** — pgvector-backed memory store with CRUD, semantic search, LLM chat, consolidation, and SSE streaming (`services/memory_service.py`, `api/routers/memories.py`)
- **SPLADE Sparse Vectors** — Learned sparse retrieval via transformers, native sparse-dense hybrid search (`services/sparse_service.py`)
- **ColBERT Multi-Vector** — Late-interaction MaxSim scoring over per-document multi-vector groups (`services/multi_vector_service.py`)
- **Natural Language Query** — English-to-structured-search via LLM (`POST /search/nl`)
- **Self-Tuning Indexes** — AI-recommended HNSW M/ef and IVF nlist/nprobe parameters (`services/index_tuner.py`)
- **Query Result Caching** — Two-tier L1 in-memory TTLCache + L2 Redis with hit-ratio monitoring (`services/query_cache.py`)
- **Streaming Search** — SSE subscriptions and webhook notifications for threshold-matched vector inserts (`services/streaming_search.py`)
- **Tiered Storage** — Hot/warm/cold auto-demotion with access-frequency tracking (`services/tiered_storage.py`)
- **Lock-Free HNSW Writes** — Thread-safe concurrent reads during ingestion (`database/hnsw_database.py`)
- **Adaptive Index Selection** — Per-query routing to fastest index based on runtime latency/recall metrics (`services/adaptive_index.py`)
- **Materialized Views** — Pre-computed common queries with auto-refresh background loop (`services/materialized_views.py`)
- **Data Retention Policies** — TTL-based vector expiry and tiered archival (`services/compliance_service.py`)
- **Query Budget Enforcement** — Per-tenant max vectors scanned, ef_search, and concurrency limits (`services/compliance_service.py`)
- **SOC2/GDPR Compliance Reports** — Auto-generated from audit logs with encryption and tenant-isolation attestation (`services/compliance_service.py`)
- **Auto Metadata Enrichment** — LLM-based entity, topic, and summary extraction on ingestion (`services/metadata_enrichment.py`)
- **Embedding Model Lifecycle** — Registry with versioning and A/B traffic splitting (`services/embedding_model_lifecycle.py`)
- **ANN Benchmark Suite** — Built-in recall@k, latency, and throughput harness with synthetic data generation (`services/benchmark_service.py`)
- **Slow Query Analyzer** — Real-time capture with p95/p99 latency, method/collection grouping (`services/slow_query_analyzer.py`)
- **Performance Middleware** — Server-Timing headers and automatic slow-query recording (`api/middleware/performance_middleware.py`)
- **Java SDK** — Maven/OkHttp/Gson client at `sdk/java/`
- **Rust SDK** — Cargo/reqwest/serde client at `sdk/rust/`
- **.NET SDK** — NuGet/HttpClient client at `sdk/dotnet/`
- **Apache Arrow Flight** — Zero-copy vector transfer via gRPC Flight protocol (`services/flight_server.py`)
- **Haystack 2.x Integration** — Document Store connector (`services/haystack_integration.py`)
- **Semantic Kernel Integration** — Memory Store connector (`services/semantic_kernel_integration.py`)
- **MCP Server** — 15 tools covering vectors, memories, sparse search, NL query, cache, tuning, slow queries (`services/mcp_server.py`)
- **Vamana/DiskANN Index** — Graph-based index with configurable L/R search parameters, streaming adjacency persistence (`database/vamana_index.py`)
- **Benchmark Snapshot** — 10K vectors, 128-dim: HNSW 0.981 recall@10, IVF 0.940, Brute 1.000, Vamana 0.970
- **Performance Dashboard** — Real-time query latency charts, slow query viewer, system health (CPU/memory/disk), enterprise compliance tab

### Changed
- **Agent Memory Layer** — migrated from standalone HNSW-based agentic_memory/ to fully integrated pgvector-backed service with MCP connectivity
- **Dashboard UI** — extended with Performance, Monitoring, Enterprise, and Integrations tabs

## [1.0.0] - 2026-06-01

### Added
- HNSW index with configurable parameters (m, ef_construction, ef_search)
- IVF index with product quantization
- KD-Tree index for low-dimensional search
- RAG pipeline with PDF/document processing
- Streaming SSE endpoint for RAG responses
- Multi-modal embedding (text, image, audio)
- Per-collection indexing
- Hybrid search (dense + sparse BM25)
- Cross-encoder reranking
- Authentication with API keys
- WebSocket streaming search
- Dashboard UI with real-time stats
- Prometheus metrics and Grafana dashboards
- Docker Compose deployment (API + PostgreSQL + Prometheus + Grafana)
- Helm chart for Kubernetes deployment
- CI/CD pipeline with GitHub Actions
- Python SDK client (`vector_db_client`)
- 112+ tests across API, index algorithms, and utilities
- 68 REST API endpoints with OpenAPI docs
- OpenAI-compatible API endpoints
- Metadata filtering (post-filter)
- Collection-scoped HNSW indexes with on-disk persistence
- PQ (Product Quantization) index
- GPU-accelerated distance kernels (optional CuPy backend)
- Bulk ingestion queue with async batching
- gRPC API (protobuf service definition + server)

### Changed
- Optimized HNSW parameters: 64% throughput improvement, 55% latency reduction, 188% recall gain
- Refactored search to support method-based dispatch (HNSW, IVF, brute, pq, hybrid)

### Documentation
- Architecture deep-dive blog posts (4-part series)
- IVF vector search guide
- Local development setup guide
- Full SDK reference in `sdk/README.md`
- Time-series vector support design doc

## [0.9.0] - 2026-01-15

### Added
- Initial HNSW and IVF indexing implementations
- PostgreSQL-backed vector storage
- Basic CRUD API
- Batch insert and search
- Benchmarking suite
- Parameter tuning guide
