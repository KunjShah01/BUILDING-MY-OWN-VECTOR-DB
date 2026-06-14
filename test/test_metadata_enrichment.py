"""Tests for metadata enrichment service."""
from __future__ import annotations
from services.metadata_enrichment import MetadataEnrichmentService


class TestMetadataEnrichmentService:
    def setup_method(self):
        self.svc = MetadataEnrichmentService()

    def test_basic_stats(self):
        result = self.svc.enrich("Hello world, email@test.com")
        assert result["_word_count"] == 3
        assert result["_char_count"] == 27

    def test_email_extraction(self):
        result = self.svc.enrich("Contact me at person@company.com or admin@test.org")
        assert len(result["_emails"]) == 2
        assert "person@company.com" in result["_emails"]

    def test_url_extraction(self):
        result = self.svc.enrich("Check https://example.com/path and http://test.org")
        assert len(result["_urls"]) == 2
