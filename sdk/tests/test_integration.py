"""Integration tests for the Python VectorDB SDK using a real HTTP server.

Uses pytest-httpserver (Python's httptest equivalent) to run a real HTTP
server on a random port, matching the patterns in the TypeScript and Go SDKs.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from werkzeug import Response as WerkzeugResponse

from vector_db_client import VectorDBClient, VectorDBHTTPError
from vector_db_client._http import raise_for_status
from vector_db_client.models import Collection, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_response(data: dict, status: int = 200) -> WerkzeugResponse:
    """Create a werkzeug JSON response for pytest-httpserver handlers."""
    return WerkzeugResponse(
        json.dumps(data), status=status, content_type="application/json"
    )


def _make_collection_response(
    col_id: str = "integ-test-col",
    name: str = "Integration",
    modality: str = "text",
    dimension: int = 384,
    embedding_model: str = "text-embedding-3-small",
) -> dict:
    return {
        "collection": {
            "collection_id": col_id,
            "name": name,
            "modality": modality,
            "dimension": dimension,
            "embedding_model": embedding_model,
        },
        "success": True,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(httpserver: HTTPServer) -> VectorDBClient:
    """Create a VectorDBClient pointed at the test server."""
    return VectorDBClient(httpserver.url_for("/"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request("/health", method="GET").respond_with_json(
            {"status": "ok"}, status=200
        )
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestCollections:
    def test_create(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections", method="POST"
        ).respond_with_json(
            _make_collection_response("demo", "Demo", "text", 384),
            status=201,
        )

        col = client.collections.create(
            "Demo",
            collection_id="demo",
            modality="text",
            dimension=384,
        )
        assert isinstance(col, Collection)
        assert col.collection_id == "demo"
        assert col.name == "Demo"
        assert col.modality == "text"
        assert col.dimension == 384
        assert col.embedding_model == "text-embedding-3-small"
        assert col.distance_metric == "cosine"

    def test_create_with_all_params(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections", method="POST"
        ).respond_with_json(
            {
                "collection": {
                    "collection_id": "full",
                    "name": "Full",
                    "modality": "image",
                    "dimension": 512,
                    "embedding_model": "clip",
                    "distance_metric": "cosine",
                    "description": "A full collection",
                },
                "success": True,
            },
            status=201,
        )

        col = client.collections.create(
            "Full",
            collection_id="full",
            modality="image",
            dimension=512,
            embedding_model="clip",
            description="A full collection",
            distance_metric="cosine",
        )
        assert col.collection_id == "full"
        assert col.dimension == 512
        assert col.description == "A full collection"

    def test_list(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections", method="GET"
        ).respond_with_json(
            {
                "collections": [
                    {
                        "collection_id": "a",
                        "name": "A",
                        "modality": "text",
                        "dimension": 128,
                        "embedding_model": "m",
                    }
                ],
                "total": 1,
            },
            status=200,
        )

        cols = client.collections.list()
        assert len(cols) == 1
        assert cols[0].collection_id == "a"
        assert cols[0].name == "A"

    def test_get(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/my-col", method="GET"
        ).respond_with_json(
            _make_collection_response("my-col", "My Col", "text", 384),
            status=200,
        )

        col = client.collections.get("my-col")
        assert col.collection_id == "my-col"
        assert col.name == "My Col"

    def test_delete(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/to-delete", method="DELETE"
        ).respond_with_json({"success": True, "deleted": True}, status=200)

        result = client.collections.delete("to-delete")
        assert result["success"] is True
        assert result["deleted"] is True

    def test_build_index(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/my-col/index", method="POST"
        ).respond_with_json(
            {"success": True, "status": "building", "method": "hnsw"}, status=200
        )

        result = client.collections.build_index(
            "my-col", method="hnsw", m=32, ef_construction=200
        )
        assert result["status"] == "building"
        assert result["method"] == "hnsw"

    def test_index_stats(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/my-col/index/stats", method="GET"
        ).respond_with_json(
            {
                "success": True,
                "vector_count": 42,
                "index_type": "hnsw",
                "status": "ready",
            },
            status=200,
        )

        stats = client.collections.index_stats("my-col")
        assert stats["vector_count"] == 42
        assert stats["status"] == "ready"


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

class TestVectors:
    def test_create(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/vectors", method="POST"
        ).respond_with_json(
            {"success": True, "vector_id": "vec-123"}, status=201
        )

        result = client.vectors.create(
            [0.1, 0.2, 0.3],
            vector_id="vec-123",
            metadata={"tag": "test"},
        )
        assert result["vector_id"] == "vec-123"

    def test_create_echoes_vector_id(self, httpserver: HTTPServer, client: VectorDBClient):
        """Server echoes back the vector_id from request body."""

        def handler(request):
            body = json.loads(request.data)
            vid = body.get("vector_id", "fallback")
            return _json_response({"success": True, "vector_id": vid})

        httpserver.expect_request(
            "/vectors", method="POST"
        ).respond_with_handler(handler)

        result = client.vectors.create([0.1], vector_id="my-custom-id")
        assert result["vector_id"] == "my-custom-id"

    def test_get(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/vectors/vec-123", method="GET"
        ).respond_with_json(
            {
                "success": True,
                "vector_id": "vec-123",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source": "test"},
            },
            status=200,
        )

        result = client.vectors.get("vec-123")
        assert result["vector_id"] == "vec-123"
        assert result["vector"] == [0.1, 0.2, 0.3]

    def test_delete(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/vectors/vec-123", method="DELETE"
        ).respond_with_json(
            {"success": True, "message": "Vector vec-123 deleted"}, status=200
        )

        result = client.vectors.delete("vec-123")
        assert result["success"] is True

    def test_search(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/search", method="POST"
        ).respond_with_json(
            {
                "success": True,
                "results": [
                    {"vector_id": "v1", "distance": 0.1, "metadata": {"label": "hit"}},
                    {"vector_id": "v2", "distance": 0.2},
                ],
                "total_results": 2,
                "search_time": 1.5,
                "method": "hnsw",
            },
            status=200,
        )

        result = client.vectors.search([0.1, 0.2, 0.3], k=5, method="hnsw")
        assert result["success"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["vector_id"] == "v1"

    def test_search_with_filters(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/search", method="POST"
        ).respond_with_json(
            {
                "success": True,
                "results": [],
                "total_results": 0,
                "search_time": 0.0,
                "method": "brute",
            },
            status=200,
        )

        result = client.vectors.search(
            [0.1], filters={"source": "web"}, k=3
        )
        assert result["total_results"] == 0


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------

class TestMultimodal:
    def test_ingest_text(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/mcol/ingest/text", method="POST"
        ).respond_with_json(
            {"success": True, "vector_id": "v_text"}, status=200
        )

        result = client.multimodal.ingest_text(
            "mcol", "hello world", metadata={"source": "test"}, vector_id="v_text"
        )
        assert result["vector_id"] == "v_text"

    def test_search_text_returns_typed_result(self, httpserver: HTTPServer, client: VectorDBClient):
        httpserver.expect_request(
            "/collections/mcol/search/text", method="POST"
        ).respond_with_json(
            {
                "success": True,
                "results": [{"vector_id": "v1", "distance": 0.05}],
                "total_results": 1,
                "search_time": 0.023,
                "method": "brute",
            },
            status=200,
        )

        result = client.multimodal.search_text("mcol", "test query", k=5)
        assert isinstance(result, SearchResult)
        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].vector_id == "v1"
        assert result.results[0].distance == 0.05
        assert result.total_results == 1
        assert result.search_time == 0.023
        assert result.method == "brute"

    def test_ingest_image_with_bytes(self, httpserver: HTTPServer, client: VectorDBClient):
        def handler(request):
            ct = request.headers.get("Content-Type", "")
            assert "multipart/form-data" in ct, "Expected multipart"
            return _json_response({"success": True, "vector_id": "v_img"})

        httpserver.expect_request(
            "/collections/mcol/ingest/image", method="POST"
        ).respond_with_handler(handler)

        result = client.multimodal.ingest_image(
            "mcol", data=b"fake-image-bytes", filename="test.jpg",
            metadata={"sku": "A1"},
        )
        assert result["vector_id"] == "v_img"

    def test_search_image_with_bytes(self, httpserver: HTTPServer, client: VectorDBClient):
        def handler(request):
            ct = request.headers.get("Content-Type", "")
            assert "multipart/form-data" in ct
            return _json_response({
                "success": True,
                "results": [{"vector_id": "v1", "distance": 0.15}],
                "total_results": 1,
                "search_time": 0.01,
                "method": "brute",
            })

        httpserver.expect_request(
            "/collections/mcol/search/image", method="POST"
        ).respond_with_handler(handler)

        result = client.multimodal.search_image("mcol", data=b"query-img", k=3)
        assert isinstance(result, SearchResult)
        assert result.results[0].distance == 0.15

    def test_ingest_audio_with_bytes(self, httpserver: HTTPServer, client: VectorDBClient):
        def handler(request):
            ct = request.headers.get("Content-Type", "")
            assert "multipart/form-data" in ct
            return _json_response({"success": True, "vector_id": "v_audio"})

        httpserver.expect_request(
            "/collections/mcol/ingest/audio", method="POST"
        ).respond_with_handler(handler)

        result = client.multimodal.ingest_audio(
            "mcol", data=b"fake-audio-bytes", vector_id="v_audio"
        )
        assert result["vector_id"] == "v_audio"

    def test_search_audio_with_bytes(self, httpserver: HTTPServer, client: VectorDBClient):
        def handler(request):
            ct = request.headers.get("Content-Type", "")
            assert "multipart/form-data" in ct
            return _json_response({
                "success": True,
                "results": [{"vector_id": "v1", "distance": 0.2}],
                "total_results": 1,
                "search_time": 0.01,
                "method": "brute",
            })

        httpserver.expect_request(
            "/collections/mcol/search/audio", method="POST"
        ).respond_with_handler(handler)

        result = client.multimodal.search_audio("mcol", data=b"query-audio", k=5)
        assert isinstance(result, SearchResult)
        assert result.results[0].distance == 0.2

    def test_ingest_image_missing_source(self, client: VectorDBClient):
        with pytest.raises(ValueError, match="Provide path or data"):
            client.multimodal.ingest_image("coll1")

    def test_search_image_missing_source(self, client: VectorDBClient):
        with pytest.raises(ValueError, match="Provide path or data"):
            client.multimodal.search_image("coll1")

    def test_ingest_audio_missing_source(self, client: VectorDBClient):
        with pytest.raises(ValueError, match="Provide path or data"):
            client.multimodal.ingest_audio("coll1")

    def test_search_audio_missing_source(self, client: VectorDBClient):
        with pytest.raises(ValueError, match="Provide path or data"):
            client.multimodal.search_audio("coll1")





# ---------------------------------------------------------------------------
# HTTP Errors
# ---------------------------------------------------------------------------

class TestHTTPErrors:
    @pytest.mark.parametrize(
        "status,detail",
        [
            (401, "Invalid API key"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (500, "Internal Server Error"),
        ],
    )
    def test_error_codes(
        self,
        httpserver: HTTPServer,
        client: VectorDBClient,
        status: int,
        detail: str,
    ):
        httpserver.expect_request(
            f"/error/{status}", method="GET"
        ).respond_with_json({"detail": detail}, status=status)

        response = client.get(f"/error/{status}")
        with pytest.raises(VectorDBHTTPError) as exc_info:
            raise_for_status(response)
        assert exc_info.value.status_code == status
        assert exc_info.value.detail == detail

    def test_401_raises_vector_db_http_error(
        self, httpserver: HTTPServer, client: VectorDBClient
    ):
        httpserver.expect_request(
            "/error/401", method="GET"
        ).respond_with_json({"detail": "Invalid API key"}, status=401)

        response = client.get("/error/401")
        with pytest.raises(VectorDBHTTPError) as exc_info:
            raise_for_status(response)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid API key"

    def test_500_without_detail_key(
        self, httpserver: HTTPServer, client: VectorDBClient
    ):
        """When response has no 'detail' key, the entire payload becomes the detail."""
        httpserver.expect_request(
            "/error/500", method="GET"
        ).respond_with_json({"error": "Internal error", "code": "ERR_001"}, status=500)

        response = client.get("/error/500")
        with pytest.raises(VectorDBHTTPError) as exc_info:
            raise_for_status(response)
        assert exc_info.value.status_code == 500
        # The entire payload becomes the detail when there's no 'detail' key
        assert exc_info.value.detail["error"] == "Internal error"


# ---------------------------------------------------------------------------
# URL encoding
# ---------------------------------------------------------------------------

class TestURLEncoding:
    def test_collection_id_with_special_characters(
        self, httpserver: HTTPServer, client: VectorDBClient
    ):
        """Collection IDs with colons and spaces should be URL-encoded."""
        httpserver.expect_request(
            "/collections/:with specials", method="GET"
        ).respond_with_json(
            {
                "collection": {
                    "collection_id": ":with specials",
                    "name": "Special Collection",
                    "modality": "text",
                    "dimension": 384,
                    "embedding_model": "default",
                },
                "success": True,
            },
            status=200,
        )

        col = client.collections.get(":with specials")
        assert col.collection_id == ":with specials"
        assert col.name == "Special Collection"


# ---------------------------------------------------------------------------
# Custom headers
# ---------------------------------------------------------------------------

class TestCustomHeaders:
    def test_client_level_headers_propagate(
        self, httpserver: HTTPServer
    ):
        """Client-level headers should be sent with every request."""
        recorded_headers = {}

        def handler(request):
            recorded_headers["authorization"] = request.headers.get("Authorization", "")
            recorded_headers["x-api-key"] = request.headers.get("X-API-Key", "")
            return _json_response({"status": "ok"})

        httpserver.expect_request(
            "/health", method="GET"
        ).respond_with_handler(handler)

        cli = VectorDBClient(
            httpserver.url_for("/"),
            headers={"Authorization": "Bearer test-token", "X-API-Key": "sk-test"},
        )
        cli.get("/health")
        assert recorded_headers["authorization"] == "Bearer test-token"
        assert recorded_headers["x-api-key"] == "sk-test"


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """End-to-end test: create collection → create vector → get → search
    → multimodal ingest → multimodal search → build index → stats → delete."""

    def test_lifecycle(self, httpserver: HTTPServer, client: VectorDBClient):
        # 1. Create collection
        httpserver.expect_request(
            "/collections", method="POST"
        ).respond_with_json(
            _make_collection_response("lifecycle-col", "Lifecycle", "text", 384),
            status=201,
        )
        col = client.collections.create(
            "Lifecycle", collection_id="lifecycle-col", dimension=384
        )
        assert col.collection_id == "lifecycle-col"

        # 2. Create vector
        def create_vec_handler(request):
            body = json.loads(request.data)
            vid = body.get("vector_id", "fallback")
            return _json_response({"success": True, "vector_id": vid})

        httpserver.expect_request(
            "/vectors", method="POST"
        ).respond_with_handler(create_vec_handler)
        vec_result = client.vectors.create([0.1, 0.2, 0.3], vector_id="lifecycle-vec")
        assert vec_result["vector_id"] == "lifecycle-vec"

        # 3. Get vector
        httpserver.expect_request(
            "/vectors/lifecycle-vec", method="GET"
        ).respond_with_json(
            {"success": True, "vector_id": "lifecycle-vec", "vector": [0.1, 0.2, 0.3]},
            status=200,
        )
        get_result = client.vectors.get("lifecycle-vec")
        assert get_result["vector_id"] == "lifecycle-vec"

        # 4. Search vectors
        httpserver.expect_request(
            "/search", method="POST"
        ).respond_with_json(
            {
                "success": True,
                "results": [{"vector_id": "match-1", "distance": 0.12}],
                "total_results": 1,
                "search_time": 0.015,
                "method": "hnsw",
            },
            status=200,
        )
        search_result = client.vectors.search([0.1], k=5)
        assert len(search_result["results"]) == 1

        # 5. Multimodal text ingest
        httpserver.expect_request(
            "/collections/lifecycle-col/ingest/text", method="POST"
        ).respond_with_json(
            {"success": True, "vector_id": "lifecycle-text"}, status=200
        )
        client.multimodal.ingest_text(
            "lifecycle-col", "Hello from lifecycle", vector_id="lifecycle-text"
        )

        # 6. Multimodal text search
        httpserver.expect_request(
            "/collections/lifecycle-col/search/text", method="POST"
        ).respond_with_json(
            {
                "success": True,
                "results": [{"vector_id": "match-1", "distance": 0.15}],
                "total_results": 1,
                "search_time": 0.023,
                "method": "brute",
            },
            status=200,
        )
        result = client.multimodal.search_text("lifecycle-col", "hello", k=5)
        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].distance == 0.15

        # 7. Build index
        httpserver.expect_request(
            "/collections/lifecycle-col/index", method="POST"
        ).respond_with_json(
            {"success": True, "status": "building", "method": "hnsw"}, status=200
        )
        client.collections.build_index("lifecycle-col", method="hnsw")

        # 8. Index stats
        httpserver.expect_request(
            "/collections/lifecycle-col/index/stats", method="GET"
        ).respond_with_json(
            {"success": True, "vector_count": 42, "index_type": "hnsw", "status": "ready"},
            status=200,
        )
        stats = client.collections.index_stats("lifecycle-col")
        assert stats["vector_count"] == 42

        # 9. Delete collection
        httpserver.expect_request(
            "/collections/lifecycle-col", method="DELETE"
        ).respond_with_json(
            {"success": True, "message": "Collection lifecycle-col deleted"}, status=200
        )
        del_result = client.collections.delete("lifecycle-col")
        assert del_result["success"] is True
