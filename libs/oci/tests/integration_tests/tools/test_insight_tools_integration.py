# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for insight tools.

These tests require real backend connections (OpenSearch, Oracle).
Skip if backends are not available.

Run with:
    cd libs/oci
    pytest tests/integration_tests/tools/test_insight_tools_integration.py -v
"""

import json
import os

import pytest

# Skip all tests if required env vars are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)


@pytest.fixture(scope="module")
def embeddings():
    """Create embedding model for tests."""
    from langchain_oci import OCIGenAIEmbeddings

    return OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
    )


@pytest.fixture(scope="module")
def opensearch_backend(embeddings):
    """Create and configure OpenSearch backend."""
    from opensearchpy import OpenSearch, RequestsHttpConnection

    from langchain_oci.tools.insight_tools import OpenSearchBackend

    host = os.environ.get("OPENSEARCH_HOST", "https://ai-dev.observ.us-ashburn-1.ocs.oraclecloud.com:9200")
    user = os.environ.get("OPENSEARCH_USER")
    password = os.environ.get("OPENSEARCH_PASSWORD")
    index_name = "unified-vector-test"

    if not user or not password:
        pytest.skip("OpenSearch credentials not configured")

    # Create test index
    client = OpenSearch(
        hosts=[host],
        http_auth=(user, password),
        use_ssl=True,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )

    # Clean up if exists
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
    except Exception:
        pass

    # Create index with k-NN enabled
    client.indices.create(
        index=index_name,
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
                    },
                }
            },
        },
    )

    backend = OpenSearchBackend(
        endpoint=host,
        index_name=index_name,
        username=user,
        password=password,
        use_ssl=True,
        verify_certs=False,
        hint="Test documents for integration testing",
    )
    backend.connect(embeddings)

    yield backend, client, index_name

    # Cleanup
    try:
        client.indices.delete(index=index_name)
    except Exception:
        pass


@pytest.fixture(scope="module")
def oracle_backend(embeddings):
    """Create and configure Oracle backend."""
    import oracledb

    from langchain_oci.tools.insight_tools import OracleBackend

    dsn = os.environ.get("ORACLE_DSN")
    user = os.environ.get("ORACLE_USER")
    password = os.environ.get("ORACLE_PASSWORD")
    wallet = os.environ.get("ORACLE_WALLET")
    table_name = "UNIFIED_VECTOR_TEST"

    if not dsn or not user or not password:
        pytest.skip("Oracle credentials not configured")

    # Create test table
    conn = oracledb.connect(
        user=user,
        password=password,
        dsn=dsn,
        config_dir=wallet,
        wallet_location=wallet,
        wallet_password=password,
    )

    cursor = conn.cursor()

    # Clean up if exists
    try:
        cursor.execute(f"DROP TABLE {table_name}")
        conn.commit()
    except Exception:
        pass

    # Create table
    cursor.execute(f"""
        CREATE TABLE {table_name} (
            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            title VARCHAR2(500),
            content CLOB,
            source VARCHAR2(200),
            embedding VECTOR(1024, FLOAT32),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()

    backend = OracleBackend(
        dsn=dsn,
        user=user,
        password=password,
        wallet_location=wallet,
        wallet_password=password,
        table_name=table_name,
        hint="Oracle test documents",
    )
    backend.connect(embeddings)

    yield backend, conn, table_name

    # Cleanup
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE {table_name}")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass


class TestOpenSearchBackendIntegration:
    """Integration tests for OpenSearch backend."""

    def test_insert_and_search(self, opensearch_backend, embeddings) -> None:
        """Test inserting and searching documents."""
        backend, client, index_name = opensearch_backend

        # Insert document
        doc_id = backend.insert(
            title="Test Document",
            content="This is a test document about Python programming.",
            source="test_integration",
            embedding=embeddings.embed_query("Python programming test"),
        )

        assert doc_id is not None
        assert len(doc_id) > 0

        # Search for it
        results = backend.search(
            query="Python programming",
            embedding=embeddings.embed_query("Python programming"),
            top_k=5,
        )

        assert len(results) >= 1
        assert any("Python" in str(r.get("content", "")) for r in results)

    def test_keyword_search(self, opensearch_backend, embeddings) -> None:
        """Test keyword search."""
        backend, client, index_name = opensearch_backend

        # Insert document with unique keyword
        backend.insert(
            title="Keyword Test",
            content="This document contains the unique keyword XYZABC123.",
            source="keyword_test",
            embedding=embeddings.embed_query("unique keyword"),
        )

        # Search by keyword
        results = backend.keyword_search("XYZABC123", top_k=5)

        assert len(results) >= 1
        assert any("XYZABC123" in str(r.get("content", "")) for r in results)

    def test_get_document(self, opensearch_backend, embeddings) -> None:
        """Test getting a document by ID."""
        backend, client, index_name = opensearch_backend

        # Insert document
        doc_id = backend.insert(
            title="Get Test",
            content="Document for get test.",
            source="get_test",
            embedding=embeddings.embed_query("get test"),
        )

        # Get it
        doc = backend.get(doc_id)

        assert doc is not None
        assert doc["id"] == doc_id
        assert doc["title"] == "Get Test"

    def test_update_document(self, opensearch_backend, embeddings) -> None:
        """Test updating a document."""
        backend, client, index_name = opensearch_backend

        # Insert document
        doc_id = backend.insert(
            title="Update Test",
            content="Original content.",
            source="update_test",
            embedding=embeddings.embed_query("update test"),
        )

        # Update it
        success = backend.update(
            document_id=doc_id,
            title="Updated Title",
            content="Updated content.",
            source=None,
            embedding=embeddings.embed_query("Updated content."),
        )

        assert success is True

        # Verify
        doc = backend.get(doc_id)
        assert doc["title"] == "Updated Title"

    def test_delete_document(self, opensearch_backend, embeddings) -> None:
        """Test deleting a document."""
        backend, client, index_name = opensearch_backend

        # Insert document
        doc_id = backend.insert(
            title="Delete Test",
            content="Document to be deleted.",
            source="delete_test",
            embedding=embeddings.embed_query("delete test"),
        )

        # Delete it
        success = backend.delete(doc_id)
        assert success is True

        # Verify it's gone
        doc = backend.get(doc_id)
        assert doc is None

    def test_bulk_insert(self, opensearch_backend, embeddings) -> None:
        """Test bulk inserting documents."""
        backend, client, index_name = opensearch_backend

        docs = [
            {"title": "Bulk 1", "content": "Bulk content 1", "source": "bulk_test"},
            {"title": "Bulk 2", "content": "Bulk content 2", "source": "bulk_test"},
            {"title": "Bulk 3", "content": "Bulk content 3", "source": "bulk_test"},
        ]
        emb_list = embeddings.embed_documents([d["content"] for d in docs])

        count = backend.bulk_insert(docs, emb_list)

        assert count == 3

    def test_stats(self, opensearch_backend) -> None:
        """Test getting stats."""
        backend, client, index_name = opensearch_backend

        stats = backend.stats()

        assert stats["backend"] == "opensearch"
        assert stats["index"] == index_name
        assert "document_count" in stats


class TestOracleBackendIntegration:
    """Integration tests for Oracle backend."""

    def test_insert_and_search(self, oracle_backend, embeddings) -> None:
        """Test inserting and searching documents."""
        backend, conn, table_name = oracle_backend

        # Insert document
        doc_id = backend.insert(
            title="Oracle Test Document",
            content="This is a test document stored in Oracle.",
            source="test_integration",
            embedding=embeddings.embed_query("Oracle test document"),
        )

        assert doc_id is not None
        assert int(doc_id) > 0

        # Search for it
        results = backend.search(
            query="Oracle test",
            embedding=embeddings.embed_query("Oracle test"),
            top_k=5,
        )

        assert len(results) >= 1

    def test_keyword_search(self, oracle_backend, embeddings) -> None:
        """Test keyword search."""
        backend, conn, table_name = oracle_backend

        # Insert document with unique keyword
        backend.insert(
            title="Oracle Keyword Test",
            content="This document contains the unique keyword ORATEST987.",
            source="keyword_test",
            embedding=embeddings.embed_query("unique keyword oracle"),
        )

        # Search by keyword
        results = backend.keyword_search("ORATEST987", top_k=5)

        assert len(results) >= 1
        assert any("ORATEST987" in str(r.get("content", "")) for r in results)

    def test_get_document(self, oracle_backend, embeddings) -> None:
        """Test getting a document by ID."""
        backend, conn, table_name = oracle_backend

        # Insert document
        doc_id = backend.insert(
            title="Oracle Get Test",
            content="Document for Oracle get test.",
            source="get_test",
            embedding=embeddings.embed_query("oracle get test"),
        )

        # Get it
        doc = backend.get(doc_id)

        assert doc is not None
        assert str(doc["id"]) == doc_id
        assert doc["title"] == "Oracle Get Test"

    def test_update_document(self, oracle_backend, embeddings) -> None:
        """Test updating a document."""
        backend, conn, table_name = oracle_backend

        # Insert document
        doc_id = backend.insert(
            title="Oracle Update Test",
            content="Original Oracle content.",
            source="update_test",
            embedding=embeddings.embed_query("oracle update test"),
        )

        # Update it
        success = backend.update(
            document_id=doc_id,
            title="Oracle Updated Title",
            content="Oracle Updated content.",
            source=None,
            embedding=embeddings.embed_query("Oracle Updated content."),
        )

        assert success is True

        # Verify
        doc = backend.get(doc_id)
        assert doc["title"] == "Oracle Updated Title"

    def test_delete_document(self, oracle_backend, embeddings) -> None:
        """Test deleting a document."""
        backend, conn, table_name = oracle_backend

        # Insert document
        doc_id = backend.insert(
            title="Oracle Delete Test",
            content="Oracle Document to be deleted.",
            source="delete_test",
            embedding=embeddings.embed_query("oracle delete test"),
        )

        # Delete it
        success = backend.delete(doc_id)
        assert success is True

        # Verify it's gone
        doc = backend.get(doc_id)
        assert doc is None

    def test_bulk_insert(self, oracle_backend, embeddings) -> None:
        """Test bulk inserting documents."""
        backend, conn, table_name = oracle_backend

        docs = [
            {"title": "Oracle Bulk 1", "content": "Oracle Bulk content 1", "source": "bulk_test"},
            {"title": "Oracle Bulk 2", "content": "Oracle Bulk content 2", "source": "bulk_test"},
            {"title": "Oracle Bulk 3", "content": "Oracle Bulk content 3", "source": "bulk_test"},
        ]
        emb_list = embeddings.embed_documents([d["content"] for d in docs])

        count = backend.bulk_insert(docs, emb_list)

        assert count == 3

    def test_stats(self, oracle_backend) -> None:
        """Test getting stats."""
        backend, conn, table_name = oracle_backend

        stats = backend.stats()

        assert stats["backend"] == "oracle"
        assert stats["table"] == table_name
        assert "document_count" in stats


class TestUnifiedToolsIntegration:
    """Integration tests for the unified tool factory."""

    def test_create_tools_with_opensearch(self, opensearch_backend, embeddings) -> None:
        """Test creating tools with OpenSearch backend."""
        from langchain_oci.tools.insight_tools import (
            OpenSearchBackend,
            create_vector_tools,
        )

        backend, client, index_name = opensearch_backend

        # Create a fresh backend instance for the factory
        new_backend = OpenSearchBackend(
            endpoint=backend.endpoint,
            index_name=index_name,
            username=backend.username,
            password=backend.password,
            use_ssl=True,
            verify_certs=False,
            hint="Logs and metrics",
        )

        tools = create_vector_tools(
            backends={"logs": new_backend},
            embedding_model=embeddings,
            enable_write=True,
            enable_delete=True,
        )

        # Should have 8 tools
        assert len(tools) == 8
        tool_names = [t.name for t in tools]
        assert "vector_search" in tool_names
        assert "keyword_search" in tool_names
        assert "get_document" in tool_names
        assert "insert_document" in tool_names
        assert "bulk_insert" in tool_names
        assert "update_document" in tool_names
        assert "delete_document" in tool_names
        assert "vector_stats" in tool_names

    def test_tool_descriptions_include_hints(self, opensearch_backend, embeddings) -> None:
        """Test tool descriptions include backend hints."""
        from langchain_oci.tools.insight_tools import (
            OpenSearchBackend,
            create_vector_tools,
        )

        backend, client, index_name = opensearch_backend

        new_backend = OpenSearchBackend(
            endpoint=backend.endpoint,
            index_name=index_name,
            username=backend.username,
            password=backend.password,
            use_ssl=True,
            verify_certs=False,
            hint="HR policies and employee benefits",
        )

        tools = create_vector_tools(
            backends={"hr": new_backend},
            embedding_model=embeddings,
        )

        search_tool = next(t for t in tools if t.name == "vector_search")
        assert "HR policies" in search_tool.description

    def test_multi_backend_setup(self, opensearch_backend, oracle_backend, embeddings) -> None:
        """Test creating tools with multiple backends."""
        from langchain_oci.tools.insight_tools import (
            OpenSearchBackend,
            OracleBackend,
            create_vector_tools,
        )

        os_backend, os_client, os_index = opensearch_backend
        ora_backend, ora_conn, ora_table = oracle_backend

        new_os_backend = OpenSearchBackend(
            endpoint=os_backend.endpoint,
            index_name=os_index,
            username=os_backend.username,
            password=os_backend.password,
            use_ssl=True,
            verify_certs=False,
            hint="Logs and monitoring data",
        )

        new_ora_backend = OracleBackend(
            dsn=ora_backend.dsn,
            user=ora_backend.user,
            password=ora_backend.password,
            wallet_location=ora_backend.wallet_location,
            table_name=ora_table,
            hint="Business transactions and reports",
        )

        tools = create_vector_tools(
            backends={"logs": new_os_backend, "business": new_ora_backend},
            default_backend="logs",
            embedding_model=embeddings,
            enable_write=True,
        )

        # Verify both backends mentioned in description
        search_tool = next(t for t in tools if t.name == "vector_search")
        assert "logs" in search_tool.description.lower()
        assert "business" in search_tool.description.lower()

    def test_source_routing(self, opensearch_backend, oracle_backend, embeddings) -> None:
        """Test source routing across backends."""
        from langchain_oci.tools.insight_tools import (
            OpenSearchBackend,
            OracleBackend,
            create_vector_tools,
        )

        os_backend, os_client, os_index = opensearch_backend
        ora_backend, ora_conn, ora_table = oracle_backend

        new_os_backend = OpenSearchBackend(
            endpoint=os_backend.endpoint,
            index_name=os_index,
            username=os_backend.username,
            password=os_backend.password,
            use_ssl=True,
            verify_certs=False,
        )

        new_ora_backend = OracleBackend(
            dsn=ora_backend.dsn,
            user=ora_backend.user,
            password=ora_backend.password,
            wallet_location=ora_backend.wallet_location,
            table_name=ora_table,
        )

        tools = create_vector_tools(
            backends={"logs": new_os_backend, "business": new_ora_backend},
            default_backend="logs",
            source_routing={
                "*_logs": "logs",
                "business_*": "business",
            },
            embedding_model=embeddings,
            enable_write=True,
        )

        # Get insert tool
        insert_tool = next(t for t in tools if t.name == "insert_document")

        # Insert with app_logs source - should go to logs
        result1 = insert_tool._run(
            title="Log Entry",
            content="Application log entry",
            source="app_logs",
        )
        assert "auto-routed" in result1.lower() or "logs" in result1.lower()

        # Insert with business_orders source - should go to business
        result2 = insert_tool._run(
            title="Order Record",
            content="Business order record",
            source="business_orders",
        )
        assert "auto-routed" in result2.lower() or "business" in result2.lower()
