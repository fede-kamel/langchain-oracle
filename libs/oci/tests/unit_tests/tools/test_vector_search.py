# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for Oracle 26ai Vector Search tools."""

from unittest.mock import MagicMock, patch


class TestOracleVectorSearchTool:
    """Test OracleVectorSearchTool."""

    def test_search_returns_results(self) -> None:
        """Test vector search returns formatted results."""
        from langchain_oci.tools.vector_search import OracleVectorSearchTool

        # Mock connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(
            return_value=iter(
                [
                    (1, "Doc 1", "Content 1", "source1", 0.2),
                    (2, "Doc 2", "Content 2", "source2", 0.3),
                ]
            )
        )
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock embedding model
        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tool = OracleVectorSearchTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
            top_k=5,
        )

        result = tool._run("test query")

        assert "Found 2 relevant documents" in result
        assert "Doc 1" in result
        assert "Doc 2" in result
        assert "similarity: 0.800" in result  # 1 - 0.2

    def test_search_no_results(self) -> None:
        """Test vector search handles no results."""
        from langchain_oci.tools.vector_search import OracleVectorSearchTool

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tool = OracleVectorSearchTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
        )

        result = tool._run("test query")

        assert "No results found" in result


class TestOracleVectorInsertTool:
    """Test OracleVectorInsertTool."""

    def test_insert_document(self) -> None:
        """Test inserting a document."""
        from langchain_oci.tools.vector_search import OracleVectorInsertTool

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tool = OracleVectorInsertTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
        )

        result = tool._run(
            title="Test Document",
            content="This is test content",
            source="test_source",
        )

        assert "Successfully inserted" in result
        assert "Test Document" in result
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class TestOracleVectorDeleteTool:
    """Test OracleVectorDeleteTool."""

    def test_delete_existing_document(self) -> None:
        """Test deleting an existing document."""
        from langchain_oci.tools.vector_search import OracleVectorDeleteTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Deleted Doc Title",)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        tool = OracleVectorDeleteTool(
            connection=mock_conn,
            table_name="TEST_VECTORS",
        )

        result = tool._run(document_id=123)

        assert "Successfully deleted" in result
        assert "Deleted Doc Title" in result
        assert mock_cursor.execute.call_count == 2  # SELECT + DELETE
        mock_conn.commit.assert_called_once()

    def test_delete_nonexistent_document(self) -> None:
        """Test deleting a document that doesn't exist."""
        from langchain_oci.tools.vector_search import OracleVectorDeleteTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        tool = OracleVectorDeleteTool(
            connection=mock_conn,
            table_name="TEST_VECTORS",
        )

        result = tool._run(document_id=999)

        assert "not found" in result


class TestOracleVectorUpdateTool:
    """Test OracleVectorUpdateTool."""

    def test_update_content(self) -> None:
        """Test updating document content regenerates embedding."""
        from langchain_oci.tools.vector_search import OracleVectorUpdateTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Old Title", "Old Content", "old_source")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.2] * 1024

        tool = OracleVectorUpdateTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
        )

        result = tool._run(
            document_id=1,
            title="New Title",
            content="New content here",
        )

        assert "Successfully updated" in result
        assert "New Title" in result
        # Should regenerate embedding since content changed
        mock_embedding.embed_query.assert_called_once_with("New content here")

    def test_update_metadata_only(self) -> None:
        """Test updating metadata without content doesn't regenerate embedding."""
        from langchain_oci.tools.vector_search import OracleVectorUpdateTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Old Title", "Old Content", "old_source")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()

        tool = OracleVectorUpdateTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
        )

        result = tool._run(
            document_id=1,
            title="New Title",
            # No content provided - should not regenerate embedding
        )

        assert "Successfully updated" in result
        mock_embedding.embed_query.assert_not_called()


class TestOracleVectorBulkInsertTool:
    """Test OracleVectorBulkInsertTool."""

    def test_bulk_insert_documents(self) -> None:
        """Test bulk inserting multiple documents."""
        from langchain_oci.tools.vector_search import OracleVectorBulkInsertTool

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]

        tool = OracleVectorBulkInsertTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
        )

        docs = [
            {"title": "Doc 1", "content": "Content 1", "source": "test"},
            {"title": "Doc 2", "content": "Content 2", "source": "test"},
        ]

        result = tool._run(documents=docs)

        assert "Successfully inserted 2 documents" in result
        mock_cursor.executemany.assert_called_once()

    def test_bulk_insert_empty_list(self) -> None:
        """Test bulk insert with empty list returns error."""
        from langchain_oci.tools.vector_search import OracleVectorBulkInsertTool

        mock_conn = MagicMock()
        mock_embedding = MagicMock()

        tool = OracleVectorBulkInsertTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
        )

        result = tool._run(documents=[])

        assert "No documents provided" in result

    def test_bulk_insert_too_many_documents(self) -> None:
        """Test bulk insert rejects more than 100 documents."""
        from langchain_oci.tools.vector_search import OracleVectorBulkInsertTool

        mock_conn = MagicMock()
        mock_embedding = MagicMock()

        tool = OracleVectorBulkInsertTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
        )

        docs = [{"title": f"Doc {i}", "content": f"Content {i}"} for i in range(101)]
        result = tool._run(documents=docs)

        assert "Maximum 100 documents" in result


class TestOracleVectorHybridSearchTool:
    """Test OracleVectorHybridSearchTool."""

    def test_hybrid_search_with_keywords(self) -> None:
        """Test hybrid search with keywords filter."""
        from langchain_oci.tools.vector_search import OracleVectorHybridSearchTool

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(
            return_value=iter(
                [
                    (1, "Doc 1", "Content with keyword", "source1", 0.15),
                ]
            )
        )
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tool = OracleVectorHybridSearchTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
            table_name="TEST_VECTORS",
        )

        result = tool._run(query="semantic query", keywords="keyword")

        assert "Found 1 relevant documents" in result
        assert "matching keywords 'keyword'" in result

    def test_hybrid_search_without_keywords(self) -> None:
        """Test hybrid search falls back to vector-only when no keywords."""
        from langchain_oci.tools.vector_search import OracleVectorHybridSearchTool

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(
            return_value=iter(
                [
                    (1, "Doc 1", "Content", "source1", 0.2),
                ]
            )
        )
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tool = OracleVectorHybridSearchTool(
            connection=mock_conn,
            embedding_model=mock_embedding,
        )

        result = tool._run(query="semantic query")

        assert "Found 1 relevant documents" in result
        assert "keywords" not in result.lower()


class TestOracleVectorGetTool:
    """Test OracleVectorGetTool."""

    def test_get_existing_document(self) -> None:
        """Test getting an existing document by ID."""
        from langchain_oci.tools.vector_search import OracleVectorGetTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1,
            "Test Title",
            "Test content here",
            "test_source",
            "2024-01-01 12:00:00",
        )
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        tool = OracleVectorGetTool(
            connection=mock_conn,
            table_name="TEST_VECTORS",
        )

        result = tool._run(document_id=1)

        assert "Document ID: 1" in result
        assert "Test Title" in result
        assert "Test content here" in result

    def test_get_nonexistent_document(self) -> None:
        """Test getting a document that doesn't exist."""
        from langchain_oci.tools.vector_search import OracleVectorGetTool

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        tool = OracleVectorGetTool(
            connection=mock_conn,
            table_name="TEST_VECTORS",
        )

        result = tool._run(document_id=999)

        assert "not found" in result


class TestOracleVectorStatsTool:
    """Test OracleVectorStatsTool."""

    def test_stats_returns_counts(self) -> None:
        """Test stats returns document counts by source."""
        from langchain_oci.tools.vector_search import OracleVectorStatsTool

        mock_cursor = MagicMock()
        # First call: total count
        # Second call: sources
        mock_cursor.fetchone.return_value = (42,)
        mock_cursor.fetchall.return_value = [
            ("source1", 20),
            ("source2", 15),
            ("source3", 7),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        tool = OracleVectorStatsTool(
            connection=mock_conn,
            table_name="TEST_VECTORS",
        )

        result = tool._run()

        assert "Total documents: 42" in result
        assert "source1: 20" in result
        assert "source2: 15" in result


class TestCreateOciVectorSearchTools:
    """Test the factory function."""

    def test_factory_creates_all_core_tools(self) -> None:
        """Test factory creates core tools."""
        import sys

        mock_oracledb = MagicMock()
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        mock_embedding = MagicMock()

        with patch.dict(sys.modules, {"oracledb": mock_oracledb}):
            from langchain_oci.tools.vector_search import create_oci_vector_search_tools

            tools = create_oci_vector_search_tools(
                dsn="test_dsn",
                user="test_user",
                password="test_pass",
                embedding_model=mock_embedding,
                enable_write=False,
                enable_delete=False,
                enable_hybrid=False,
            )

            # Should have: search, get, stats (3 core tools)
            assert len(tools) == 3
            tool_names = [t.name for t in tools]
            assert "vector_search" in tool_names
            assert "vector_get" in tool_names
            assert "vector_stats" in tool_names

    def test_factory_with_write_enabled(self) -> None:
        """Test factory includes write tools when enabled."""
        import sys

        mock_oracledb = MagicMock()
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        mock_embedding = MagicMock()

        with patch.dict(sys.modules, {"oracledb": mock_oracledb}):
            from langchain_oci.tools.vector_search import create_oci_vector_search_tools

            tools = create_oci_vector_search_tools(
                dsn="test_dsn",
                user="test_user",
                password="test_pass",
                embedding_model=mock_embedding,
                enable_write=True,
                enable_delete=False,
                enable_hybrid=False,
            )

            # Should have: search, get, stats + insert, update, bulk_insert
            assert len(tools) == 6
            tool_names = [t.name for t in tools]
            assert "vector_insert" in tool_names
            assert "vector_update" in tool_names
            assert "vector_bulk_insert" in tool_names

    def test_factory_with_delete_enabled(self) -> None:
        """Test factory includes delete tool when enabled."""
        import sys

        mock_oracledb = MagicMock()
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        mock_embedding = MagicMock()

        with patch.dict(sys.modules, {"oracledb": mock_oracledb}):
            from langchain_oci.tools.vector_search import create_oci_vector_search_tools

            tools = create_oci_vector_search_tools(
                dsn="test_dsn",
                user="test_user",
                password="test_pass",
                embedding_model=mock_embedding,
                enable_write=False,
                enable_delete=True,
                enable_hybrid=False,
            )

            # Should have: search, get, stats + delete
            assert len(tools) == 4
            tool_names = [t.name for t in tools]
            assert "vector_delete" in tool_names

    def test_factory_with_hybrid_enabled(self) -> None:
        """Test factory includes hybrid search tool when enabled."""
        import sys

        mock_oracledb = MagicMock()
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        mock_embedding = MagicMock()

        with patch.dict(sys.modules, {"oracledb": mock_oracledb}):
            from langchain_oci.tools.vector_search import create_oci_vector_search_tools

            tools = create_oci_vector_search_tools(
                dsn="test_dsn",
                user="test_user",
                password="test_pass",
                embedding_model=mock_embedding,
                enable_write=False,
                enable_delete=False,
                enable_hybrid=True,
            )

            # Should have: search, get, stats + hybrid_search
            assert len(tools) == 4
            tool_names = [t.name for t in tools]
            assert "vector_hybrid_search" in tool_names

    def test_factory_all_enabled(self) -> None:
        """Test factory creates all tools when everything enabled."""
        import sys

        mock_oracledb = MagicMock()
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        mock_embedding = MagicMock()

        with patch.dict(sys.modules, {"oracledb": mock_oracledb}):
            from langchain_oci.tools.vector_search import create_oci_vector_search_tools

            tools = create_oci_vector_search_tools(
                dsn="test_dsn",
                user="test_user",
                password="test_pass",
                embedding_model=mock_embedding,
                enable_write=True,
                enable_delete=True,
                enable_hybrid=True,
            )

            # All 8 tools
            assert len(tools) == 8
            tool_names = [t.name for t in tools]
            assert "vector_search" in tool_names
            assert "vector_get" in tool_names
            assert "vector_stats" in tool_names
            assert "vector_insert" in tool_names
            assert "vector_update" in tool_names
            assert "vector_bulk_insert" in tool_names
            assert "vector_hybrid_search" in tool_names
            assert "vector_delete" in tool_names
