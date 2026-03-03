# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for insight tools (auto-routing data search)."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestBackendRouter:
    """Test BackendRouter routing logic."""

    def test_route_by_source_exact_match(self) -> None:
        """Test exact source match."""
        from langchain_oci.tools.insight_tools import BackendRouter

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        mock_backend = MagicMock()
        mock_backend.hint = "test"

        router = BackendRouter(
            backends={"logs": mock_backend, "business": mock_backend},
            embedding_model=mock_embedding,
            default_backend="logs",
            source_routing={"app_logs": "logs", "orders": "business"},
        )

        assert router.route_by_source("app_logs") == "logs"
        assert router.route_by_source("orders") == "business"
        assert router.route_by_source("unknown") == "logs"  # default

    def test_route_by_source_wildcard(self) -> None:
        """Test wildcard source patterns."""
        from langchain_oci.tools.insight_tools import BackendRouter

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        mock_backend = MagicMock()
        mock_backend.hint = "test"

        router = BackendRouter(
            backends={"logs": mock_backend, "business": mock_backend},
            embedding_model=mock_embedding,
            default_backend="logs",
            source_routing={"*_logs": "logs", "business_*": "business"},
        )

        # Suffix wildcard
        assert router.route_by_source("app_logs") == "logs"
        assert router.route_by_source("system_logs") == "logs"

        # Prefix wildcard
        assert router.route_by_source("business_orders") == "business"
        assert router.route_by_source("business_finance") == "business"

    def test_route_by_source_explicit_override(self) -> None:
        """Test explicit backend overrides routing."""
        from langchain_oci.tools.insight_tools import BackendRouter

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        mock_backend = MagicMock()
        mock_backend.hint = "test"

        router = BackendRouter(
            backends={"logs": mock_backend, "business": mock_backend},
            embedding_model=mock_embedding,
            default_backend="logs",
            source_routing={"*_logs": "logs"},
        )

        # Explicit override takes precedence
        assert router.route_by_source("app_logs", explicit_backend="business") == "business"

    def test_single_backend_no_routing_needed(self) -> None:
        """Test single backend always returns that backend."""
        from langchain_oci.tools.insight_tools import BackendRouter

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        mock_backend = MagicMock()
        mock_backend.hint = "test"

        router = BackendRouter(
            backends={"only": mock_backend},
            embedding_model=mock_embedding,
            default_backend="only",
        )

        assert router.route_query("any query") == "only"


class TestOpenSearchBackend:
    """Test OpenSearchBackend."""

    def test_backend_name(self) -> None:
        """Test backend name property."""
        from langchain_oci.tools.insight_tools import OpenSearchBackend

        backend = OpenSearchBackend(
            endpoint="https://test.com",
            index_name="test-index",
        )
        assert backend.name == "opensearch"

    def test_backend_hint(self) -> None:
        """Test backend hint property."""
        from langchain_oci.tools.insight_tools import OpenSearchBackend

        backend = OpenSearchBackend(
            endpoint="https://test.com",
            index_name="test-index",
            hint="HR policies and benefits",
        )
        assert backend.hint == "HR policies and benefits"

    def test_search_returns_results(self) -> None:
        """Test search returns results."""
        from langchain_oci.tools.insight_tools import OpenSearchBackend

        backend = OpenSearchBackend(
            endpoint="https://test.com",
            index_name="test-index",
        )

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "1", "_score": 0.95, "_source": {"title": "Doc 1", "content": "Content 1"}},
                    {"_id": "2", "_score": 0.85, "_source": {"title": "Doc 2", "content": "Content 2"}},
                ]
            }
        }
        backend._client = mock_client

        results = backend.search("test query", [0.1] * 1024, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95

    def test_insert_document(self) -> None:
        """Test document insertion."""
        from langchain_oci.tools.insight_tools import OpenSearchBackend

        backend = OpenSearchBackend(
            endpoint="https://test.com",
            index_name="test-index",
        )

        mock_client = MagicMock()
        mock_client.index.return_value = {"_id": "new-doc-id"}
        backend._client = mock_client

        doc_id = backend.insert("Title", "Content", "test_source", [0.1] * 1024)

        assert doc_id == "new-doc-id"

    def test_delete_document(self) -> None:
        """Test document deletion."""
        from langchain_oci.tools.insight_tools import OpenSearchBackend

        backend = OpenSearchBackend(
            endpoint="https://test.com",
            index_name="test-index",
        )

        mock_client = MagicMock()
        mock_client.delete.return_value = {"result": "deleted"}
        backend._client = mock_client

        success = backend.delete("doc-id")
        assert success is True


class TestOracleBackend:
    """Test OracleBackend."""

    def test_backend_name(self) -> None:
        """Test backend name property."""
        from langchain_oci.tools.insight_tools import OracleBackend

        backend = OracleBackend(
            dsn="test_db",
            user="admin",
            password="secret",
        )
        assert backend.name == "oracle"

    def test_backend_hint(self) -> None:
        """Test backend hint property."""
        from langchain_oci.tools.insight_tools import OracleBackend

        backend = OracleBackend(
            dsn="test_db",
            user="admin",
            password="secret",
            hint="Business transactions",
        )
        assert backend.hint == "Business transactions"

    def test_search_returns_results(self) -> None:
        """Test search returns results."""
        from langchain_oci.tools.insight_tools import OracleBackend

        backend = OracleBackend(
            dsn="test_db",
            user="admin",
            password="secret",
        )

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(
            return_value=iter([
                (1, "Doc 1", "Content 1", "source1", 0.1),
                (2, "Doc 2", "Content 2", "source2", 0.2),
            ])
        )
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._connection = mock_conn

        results = backend.search("test", [0.1] * 1024, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.9  # 1 - 0.1 distance

    def test_insert_returns_id(self) -> None:
        """Test insert returns document ID."""
        from langchain_oci.tools.insight_tools import OracleBackend

        backend = OracleBackend(
            dsn="test_db",
            user="admin",
            password="secret",
        )

        mock_cursor = MagicMock()
        mock_out_var = MagicMock()
        mock_out_var.getvalue.return_value = [42]
        mock_cursor.var.return_value = mock_out_var

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._connection = mock_conn

        doc_id = backend.insert("Title", "Content", "source", [0.1] * 1024)

        assert doc_id == "42"


class TestInsightSearchTool:
    """Test InsightSearchTool with auto-routing."""

    def test_search_auto_routes(self) -> None:
        """Test search auto-routes to best backend."""
        from langchain_oci.tools.insight_tools import (
            BackendRouter,
            InsightSearchTool,
            OpenSearchBackend,
        )

        # Create backend with mocked client
        backend = OpenSearchBackend(endpoint="https://test.com", index_name="test")
        backend._client = MagicMock()
        backend._client.search.return_value = {
            "hits": {"hits": [{"_id": "1", "_score": 0.9, "_source": {"title": "Result", "content": "Content"}}]}
        }

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        router = BackendRouter(
            backends={"hr": backend},
            embedding_model=mock_embedding,
            default_backend="hr",
        )

        tool = InsightSearchTool(router=router, top_k=5)

        # No backend param - should auto-route
        result = tool._run("vacation policy")

        assert "Result" in result
        backend._client.search.assert_called_once()


class TestInsightInsertTool:
    """Test InsightInsertTool with source routing."""

    def test_insert_routes_by_source(self) -> None:
        """Test insert routes based on source."""
        from langchain_oci.tools.insight_tools import (
            BackendRouter,
            InsightInsertTool,
            OpenSearchBackend,
        )

        backend1 = OpenSearchBackend(endpoint="https://test.com", index_name="logs")
        backend1._client = MagicMock()
        backend1._client.index.return_value = {"_id": "logs-id"}

        backend2 = OpenSearchBackend(endpoint="https://test.com", index_name="business")
        backend2._client = MagicMock()
        backend2._client.index.return_value = {"_id": "business-id"}

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        router = BackendRouter(
            backends={"logs": backend1, "business": backend2},
            embedding_model=mock_embedding,
            default_backend="logs",
            source_routing={"*_logs": "logs", "business_*": "business"},
        )

        tool = InsightInsertTool(router=router)

        # Should route to logs
        result1 = tool._run("Title", "Content", "app_logs")
        assert "logs" in result1.lower()

        # Should route to business
        result2 = tool._run("Title", "Content", "business_orders")
        assert "business" in result2.lower()


class TestCreateInsightTools:
    """Test the factory function."""

    def test_requires_at_least_one_backend(self) -> None:
        """Test factory requires at least one backend."""
        from langchain_oci.tools.insight_tools import create_insight_tools

        with pytest.raises(ValueError, match="At least one backend"):
            create_insight_tools(backends={})

    def test_validates_default_backend(self) -> None:
        """Test factory validates default backend exists."""
        import sys

        mock_opensearch = MagicMock()
        mock_client = MagicMock()
        mock_opensearch.OpenSearch.return_value = mock_client

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.insight_tools import (
                OpenSearchBackend,
                create_insight_tools,
            )

            backend = OpenSearchBackend(
                endpoint="https://test.com",
                index_name="test",
            )

            with pytest.raises(ValueError, match="not in backends"):
                create_insight_tools(
                    backends={"logs": backend},
                    default_backend="nonexistent",
                    embedding_model=MagicMock(),
                )

    def test_creates_core_tools(self) -> None:
        """Test factory creates core tools."""
        import sys

        mock_opensearch = MagicMock()
        mock_client = MagicMock()
        mock_opensearch.OpenSearch.return_value = mock_client

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.insight_tools import (
                OpenSearchBackend,
                create_insight_tools,
            )

            backend = OpenSearchBackend(
                endpoint="https://test.com",
                index_name="test",
            )

            mock_embedding = MagicMock()
            mock_embedding.embed_query.return_value = [0.1] * 1024

            tools = create_insight_tools(
                backends={"logs": backend},
                embedding_model=mock_embedding,
                enable_write=False,
                enable_delete=False,
            )

            # Core tools: search, keyword_search, get_document, stats
            assert len(tools) == 4
            tool_names = [t.name for t in tools]
            assert "search" in tool_names
            assert "keyword_search" in tool_names
            assert "get_document" in tool_names
            assert "stats" in tool_names

    def test_creates_write_tools_when_enabled(self) -> None:
        """Test factory creates write tools when enabled."""
        import sys

        mock_opensearch = MagicMock()
        mock_client = MagicMock()
        mock_opensearch.OpenSearch.return_value = mock_client

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.insight_tools import (
                OpenSearchBackend,
                create_insight_tools,
            )

            backend = OpenSearchBackend(
                endpoint="https://test.com",
                index_name="test",
            )

            mock_embedding = MagicMock()
            mock_embedding.embed_query.return_value = [0.1] * 1024

            tools = create_insight_tools(
                backends={"logs": backend},
                embedding_model=mock_embedding,
                enable_write=True,
                enable_delete=False,
            )

            # Core + write tools
            assert len(tools) == 7
            tool_names = [t.name for t in tools]
            assert "insert_document" in tool_names
            assert "bulk_insert" in tool_names
            assert "update_document" in tool_names

    def test_creates_delete_tool_when_enabled(self) -> None:
        """Test factory creates delete tool when enabled."""
        import sys

        mock_opensearch = MagicMock()
        mock_client = MagicMock()
        mock_opensearch.OpenSearch.return_value = mock_client

        with patch.dict(sys.modules, {"opensearchpy": mock_opensearch}):
            from langchain_oci.tools.insight_tools import (
                OpenSearchBackend,
                create_insight_tools,
            )

            backend = OpenSearchBackend(
                endpoint="https://test.com",
                index_name="test",
            )

            mock_embedding = MagicMock()
            mock_embedding.embed_query.return_value = [0.1] * 1024

            tools = create_insight_tools(
                backends={"logs": backend},
                embedding_model=mock_embedding,
                enable_write=False,
                enable_delete=True,
            )

            # Core + delete
            assert len(tools) == 5
            tool_names = [t.name for t in tools]
            assert "delete_document" in tool_names


class TestArgSchemas:
    """Test argument schemas."""

    def test_search_args(self) -> None:
        """Test SearchArgs schema."""
        from langchain_oci.tools.insight_tools import SearchArgs

        args = SearchArgs(query="test query")
        assert args.query == "test query"

    def test_insert_document_args(self) -> None:
        """Test InsertDocumentArgs schema."""
        from langchain_oci.tools.insight_tools import InsertDocumentArgs

        args = InsertDocumentArgs(
            title="Test Title",
            content="Test content",
            source="test_source",
        )
        assert args.title == "Test Title"
        assert args.content == "Test content"
        assert args.source == "test_source"

    def test_insert_document_args_defaults(self) -> None:
        """Test InsertDocumentArgs default values."""
        from langchain_oci.tools.insight_tools import InsertDocumentArgs

        args = InsertDocumentArgs(title="Title", content="Content")
        assert args.source == "user_input"

    def test_stats_args(self) -> None:
        """Test StatsArgs schema."""
        from langchain_oci.tools.insight_tools import StatsArgs

        args = StatsArgs(backend="all")
        assert args.backend == "all"

        args_default = StatsArgs()
        assert args_default.backend is None
