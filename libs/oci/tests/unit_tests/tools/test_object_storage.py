# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Object Storage tools."""

from unittest.mock import MagicMock, patch

from langchain_oci.tools.object_storage import (
    OCIObjectStorageListTool,
    OCIObjectStorageReadTextTool,
    OCIObjectStorageReadTool,
    OCIObjectStorageSearchTool,
    OCIObjectStorageWriteTool,
)


class TestOCIObjectStorageListTool:
    """Tests for OCIObjectStorageListTool."""

    def test_list_objects_success(self) -> None:
        """Test listing objects in a bucket."""
        mock_client = MagicMock()
        mock_obj = MagicMock()
        mock_obj.name = "data/file1.json"
        mock_obj.size = 1024

        mock_response = MagicMock()
        mock_response.data.objects = [mock_obj]
        mock_client.list_objects.return_value = mock_response

        tool = OCIObjectStorageListTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket")
        assert "file1.json" in result
        assert "1.0 KB" in result

    def test_list_objects_unauthorized_bucket(self) -> None:
        """Test error when bucket not in allowed list."""
        mock_client = MagicMock()
        tool = OCIObjectStorageListTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["allowed-bucket"],
        )

        result = tool._run("unauthorized-bucket")
        assert "Error" in result
        assert "not in allowed list" in result


class TestOCIObjectStorageReadTool:
    """Tests for OCIObjectStorageReadTool."""

    def test_read_json_success(self) -> None:
        """Test reading a JSON file."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data.content = b'{"key": "value"}'
        mock_client.get_object.return_value = mock_response

        tool = OCIObjectStorageReadTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket", "data.json")
        assert "key" in result
        assert "value" in result


class TestOCIObjectStorageReadTextTool:
    """Tests for OCIObjectStorageReadTextTool."""

    def test_read_text_success(self) -> None:
        """Test reading a text file."""
        mock_client = MagicMock()

        # Mock head_object for size check
        mock_head = MagicMock()
        mock_head.headers = {"content-length": "1024"}
        mock_client.head_object.return_value = mock_head

        # Mock get_object for content
        mock_response = MagicMock()
        mock_response.data.content = b"Hello, this is a text file."
        mock_client.get_object.return_value = mock_response

        tool = OCIObjectStorageReadTextTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket", "readme.txt")
        assert "Hello" in result

    def test_read_csv_formats_as_table(self) -> None:
        """Test reading a CSV file formats as markdown table."""
        mock_client = MagicMock()

        mock_head = MagicMock()
        mock_head.headers = {"content-length": "100"}
        mock_client.head_object.return_value = mock_head

        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        mock_response = MagicMock()
        mock_response.data.content = csv_content.encode("utf-8")
        mock_client.get_object.return_value = mock_response

        tool = OCIObjectStorageReadTextTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket", "data.csv")
        assert "name" in result
        assert "Alice" in result
        assert "|" in result  # Markdown table format

    def test_rejects_large_files(self) -> None:
        """Test that large files are rejected."""
        mock_client = MagicMock()

        mock_head = MagicMock()
        mock_head.headers = {"content-length": str(1024 * 1024)}  # 1MB
        mock_client.head_object.return_value = mock_head

        tool = OCIObjectStorageReadTextTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
            max_size_kb=500,
        )

        result = tool._run("test-bucket", "huge.txt")
        assert "too large" in result


class TestOCIObjectStorageWriteTool:
    """Tests for OCIObjectStorageWriteTool."""

    def test_write_success(self) -> None:
        """Test writing a file."""
        mock_client = MagicMock()

        tool = OCIObjectStorageWriteTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket", "output/result.json", '{"result": 42}')
        assert "Successfully wrote" in result
        mock_client.put_object.assert_called_once()

    def test_write_unauthorized_bucket(self) -> None:
        """Test error when bucket not in allowed list."""
        mock_client = MagicMock()

        tool = OCIObjectStorageWriteTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["allowed-bucket"],
        )

        result = tool._run("unauthorized-bucket", "file.txt", "content")
        assert "Error" in result
        assert "not in allowed list" in result

    def test_content_type_detection(self) -> None:
        """Test that content type is detected from file extension."""
        mock_client = MagicMock()

        tool = OCIObjectStorageWriteTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        # Write JSON
        tool._run("test-bucket", "data.json", "{}")
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["content_type"] == "application/json"

        # Write CSV
        tool._run("test-bucket", "data.csv", "a,b,c")
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["content_type"] == "text/csv"

        # Write Markdown
        tool._run("test-bucket", "report.md", "# Report")
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["content_type"] == "text/markdown"


class TestOCIObjectStorageSearchTool:
    """Tests for OCIObjectStorageSearchTool."""

    def test_search_finds_matches(self) -> None:
        """Test searching finds matching content."""
        mock_client = MagicMock()

        # Mock list_objects
        mock_obj = MagicMock()
        mock_obj.name = "data.json"
        mock_list_response = MagicMock()
        mock_list_response.data.objects = [mock_obj]
        mock_client.list_objects.return_value = mock_list_response

        # Mock get_object
        mock_response = MagicMock()
        mock_response.data.content = b'[{"topic": "AI research", "content": "ML"}]'
        mock_client.get_object.return_value = mock_response

        tool = OCIObjectStorageSearchTool(
            client=mock_client,
            namespace="test-ns",
            buckets=["test-bucket"],
        )

        result = tool._run("test-bucket", "AI")
        assert "Found" in result or "AI" in result


class TestCreateOCIObjectStorageTools:
    """Tests for create_oci_object_storage_tools factory function."""

    def test_creates_all_tools(self) -> None:
        """Test that factory creates all expected tools."""
        import sys

        # Create mock oci module
        mock_oci = MagicMock()
        mock_config = {"region": "us-ashburn-1"}
        mock_oci.config.from_file.return_value = mock_config
        mock_oci.object_storage.ObjectStorageClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"oci": mock_oci}):
            from langchain_oci.tools.object_storage import (
                create_oci_object_storage_tools,
            )

            tools = create_oci_object_storage_tools(
                namespace="test-ns",
                buckets=["bucket1"],
                enable_write=True,
                enable_copy=True,
            )

            # Should have 7 tools: list, read, read_text, info, search, write, copy
            assert len(tools) == 7
            tool_names = [t.name for t in tools]
            assert "list_bucket_objects" in tool_names
            assert "read_bucket_object" in tool_names
            assert "read_text_file" in tool_names
            assert "get_object_info" in tool_names
            assert "search_bucket_data" in tool_names
            assert "write_bucket_object" in tool_names
            assert "copy_bucket_object" in tool_names

    def test_disable_write(self) -> None:
        """Test that write tool can be disabled."""
        import sys

        mock_oci = MagicMock()
        mock_config = {"region": "us-ashburn-1"}
        mock_oci.config.from_file.return_value = mock_config
        mock_oci.object_storage.ObjectStorageClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"oci": mock_oci}):
            from langchain_oci.tools.object_storage import (
                create_oci_object_storage_tools,
            )

            tools = create_oci_object_storage_tools(
                namespace="test-ns",
                buckets=["bucket1"],
                enable_write=False,
                enable_copy=False,
            )

            # Should have 5 core tools (no write, no copy)
            assert len(tools) == 5
            tool_names = [t.name for t in tools]
            assert "write_bucket_object" not in tool_names
            assert "copy_bucket_object" not in tool_names
