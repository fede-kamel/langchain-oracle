# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI OpenSearch tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_oci.tools.opensearch import (
    OCIOpenSearchBulkInsertTool,
    OCIOpenSearchDeleteTool,
    OCIOpenSearchHybridSearchTool,
    OCIOpenSearchInsertTool,
    OCIOpenSearchKeywordSearchTool,
    OCIOpenSearchStatsTool,
    OCIOpenSearchVectorSearchTool,
)


@pytest.fixture
def mock_client():
    """Create a mock OpenSearch client."""
    return MagicMock()


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = MagicMock()
    model.embed_query.return_value = [0.1] * 1024
    model.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    return model


class TestOCIOpenSearchVectorSearchTool:
    """Tests for OCIOpenSearchVectorSearchTool."""

    def test_vector_search_success(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test successful vector search."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "title": "AI Research",
                            "content": "Machine learning advances...",
                            "source": "papers",
                        },
                    }
                ]
            }
        }

        tool = OCIOpenSearchVectorSearchTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run("machine learning")
        assert "AI Research" in result
        assert "0.95" in result
        mock_embedding_model.embed_query.assert_called_once_with("machine learning")

    def test_vector_search_no_results(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test vector search with no results."""
        mock_client.search.return_value = {"hits": {"hits": []}}

        tool = OCIOpenSearchVectorSearchTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run("nonexistent topic")
        assert "No results found" in result


class TestOCIOpenSearchKeywordSearchTool:
    """Tests for OCIOpenSearchKeywordSearchTool."""

    def test_keyword_search_success(self, mock_client: MagicMock) -> None:
        """Test successful keyword search."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 8.5,
                        "_source": {
                            "title": "Python Tutorial",
                            "content": "Learn Python programming...",
                        },
                    }
                ]
            }
        }

        tool = OCIOpenSearchKeywordSearchTool(
            client=mock_client,
            index_name="test-index",
        )

        result = tool._run("Python programming")
        assert "Python Tutorial" in result
        assert "8.5" in result

    def test_keyword_search_no_results(self, mock_client: MagicMock) -> None:
        """Test keyword search with no results."""
        mock_client.search.return_value = {"hits": {"hits": []}}

        tool = OCIOpenSearchKeywordSearchTool(
            client=mock_client,
            index_name="test-index",
        )

        result = tool._run("xyz123nonexistent")
        assert "No results found" in result


class TestOCIOpenSearchHybridSearchTool:
    """Tests for OCIOpenSearchHybridSearchTool."""

    def test_hybrid_search_success(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test successful hybrid search."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            "title": "Deep Learning Guide",
                            "content": "Neural networks explained...",
                        },
                    }
                ]
            }
        }

        tool = OCIOpenSearchHybridSearchTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run("neural networks")
        assert "Deep Learning Guide" in result
        assert "hybrid search" in result.lower() or "combined" in result.lower()


class TestOCIOpenSearchInsertTool:
    """Tests for OCIOpenSearchInsertTool."""

    def test_insert_success(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test successful document insertion."""
        mock_client.index.return_value = {"_id": "doc123", "result": "created"}

        tool = OCIOpenSearchInsertTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run(
            title="Test Document",
            content="This is test content.",
            source="unit_test",
        )

        assert "Successfully inserted" in result
        assert "doc123" in result
        mock_client.index.assert_called_once()

    def test_insert_with_metadata(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test insertion with metadata."""
        mock_client.index.return_value = {"_id": "doc456", "result": "created"}

        tool = OCIOpenSearchInsertTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run(
            title="Doc with Meta",
            content="Content here.",
            metadata={"category": "test", "priority": 1},
        )

        assert "Successfully inserted" in result
        # Verify metadata was included in the call
        call_args = mock_client.index.call_args
        assert "metadata" in call_args.kwargs["body"]


class TestOCIOpenSearchBulkInsertTool:
    """Tests for OCIOpenSearchBulkInsertTool."""

    def test_bulk_insert_success(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test successful bulk insertion."""
        mock_client.bulk.return_value = {"errors": False, "items": [{}, {}]}

        tool = OCIOpenSearchBulkInsertTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]

        result = tool._run(json.dumps(docs))
        assert "Successfully inserted 2 documents" in result

    def test_bulk_insert_with_errors(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test bulk insertion with some errors."""
        mock_client.bulk.return_value = {
            "errors": True,
            "items": [{"index": {}}, {"index": {"error": "failed"}}],
        }

        tool = OCIOpenSearchBulkInsertTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]

        result = tool._run(json.dumps(docs))
        assert "1 errors" in result

    def test_bulk_insert_invalid_json(
        self, mock_client: MagicMock, mock_embedding_model: MagicMock
    ) -> None:
        """Test bulk insertion with invalid JSON."""
        tool = OCIOpenSearchBulkInsertTool(
            client=mock_client,
            embedding_model=mock_embedding_model,
            index_name="test-index",
        )

        result = tool._run("not valid json")
        assert "Invalid JSON" in result


class TestOCIOpenSearchDeleteTool:
    """Tests for OCIOpenSearchDeleteTool."""

    def test_delete_success(self, mock_client: MagicMock) -> None:
        """Test successful deletion."""
        mock_client.delete.return_value = {"result": "deleted"}

        tool = OCIOpenSearchDeleteTool(
            client=mock_client,
            index_name="test-index",
        )

        result = tool._run("doc123")
        assert "Successfully deleted" in result

    def test_delete_not_found(self, mock_client: MagicMock) -> None:
        """Test deletion of non-existent document."""
        mock_client.delete.return_value = {"result": "not_found"}

        tool = OCIOpenSearchDeleteTool(
            client=mock_client,
            index_name="test-index",
        )

        result = tool._run("nonexistent")
        assert "not found" in result.lower()


class TestOCIOpenSearchStatsTool:
    """Tests for OCIOpenSearchStatsTool."""

    def test_stats_success(self, mock_client: MagicMock) -> None:
        """Test successful stats retrieval."""
        mock_client.indices.stats.return_value = {
            "indices": {
                "test-index": {
                    "primaries": {
                        "docs": {"count": 1000},
                        "store": {"size_in_bytes": 10485760},  # 10 MB
                    }
                }
            }
        }
        mock_client.indices.get_mapping.return_value = {
            "test-index": {
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "embedding": {"type": "knn_vector"},
                    }
                }
            }
        }
        # Mock aggregation for sources
        mock_client.search.return_value = {
            "aggregations": {
                "sources": {
                    "buckets": [
                        {"key": "papers", "doc_count": 500},
                        {"key": "web", "doc_count": 300},
                    ]
                }
            }
        }

        tool = OCIOpenSearchStatsTool(
            client=mock_client,
            index_name="test-index",
        )

        result = tool._run()
        assert "1,000" in result  # Document count
        assert "10" in result  # Size in MB
        assert "title" in result or "content" in result  # Fields


class TestCreateOCIOpenSearchTools:
    """Tests for create_oci_opensearch_tools factory function."""

    def test_creates_all_tools(self) -> None:
        """Test that factory creates all expected tools."""
        import sys

        mock_opensearch = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.info.return_value = {
            "version": {"number": "2.11.0"},
            "cluster_name": "test",
        }
        mock_opensearch.OpenSearch.return_value = mock_client_instance

        # Create a mock embedding model to pass directly
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1024

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.opensearch import create_oci_opensearch_tools

            tools = create_oci_opensearch_tools(
                endpoint="https://localhost:9200",
                index_name="test-index",
                username="admin",
                password="admin",
                embedding_model=mock_embedding_model,  # Provide directly
                enable_write=True,
                enable_delete=True,
                enable_hybrid=True,
            )

            # Should have 7 tools: vector, keyword, hybrid, stats,
            # insert, bulk_insert, delete
            assert len(tools) == 7
            tool_names = [t.name for t in tools]
            assert "opensearch_vector_search" in tool_names
            assert "opensearch_keyword_search" in tool_names
            assert "opensearch_hybrid_search" in tool_names
            assert "opensearch_stats" in tool_names
            assert "opensearch_insert" in tool_names
            assert "opensearch_bulk_insert" in tool_names
            assert "opensearch_delete" in tool_names

    def test_disable_optional_tools(self) -> None:
        """Test that optional tools can be disabled."""
        import sys

        mock_opensearch = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.info.return_value = {"version": {}, "cluster_name": ""}
        mock_opensearch.OpenSearch.return_value = mock_client_instance

        mock_embedding_model = MagicMock()

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.opensearch import create_oci_opensearch_tools

            tools = create_oci_opensearch_tools(
                endpoint="https://localhost:9200",
                index_name="test-index",
                username="admin",
                password="admin",
                embedding_model=mock_embedding_model,
                enable_write=False,
                enable_delete=False,
                enable_hybrid=False,
            )

            # Should have 3 tools: vector, keyword, stats
            assert len(tools) == 3
            tool_names = [t.name for t in tools]
            assert "opensearch_vector_search" in tool_names
            assert "opensearch_keyword_search" in tool_names
            assert "opensearch_stats" in tool_names
            assert "opensearch_hybrid_search" not in tool_names
            assert "opensearch_insert" not in tool_names
