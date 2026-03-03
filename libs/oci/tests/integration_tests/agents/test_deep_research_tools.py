# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for Deep Agent with OCI Tools.

These tests verify that the deep agent can use OCI tools for actual
deep research tasks. Each tool set requires specific OCI resources.

## Prerequisites

### For Object Storage Tools
```bash
export OCI_NAMESPACE="your-namespace"
export OCI_BUCKET="your-bucket-name"
export OCI_CONFIG_PROFILE="API_KEY_AUTH"
```

### For OpenSearch Tools
```bash
export OPENSEARCH_HOST="https://your-opensearch.us-ashburn-1.oci.oraclecloud.com"
export OPENSEARCH_INDEX="research-vectors"
export OPENSEARCH_USER="admin"
export OPENSEARCH_PASSWORD="your-password"
```

### For Oracle Vector Search Tools (26ai)
```bash
export ORACLE_DSN="deepresearch_low"
export ORACLE_USER="ADMIN"
export ORACLE_PASSWORD="your-password"
export ORACLE_WALLET_LOCATION="~/.oracle-wallet/deepresearch"
```

## Running Tests

```bash
# Run all deep agent tools tests
pytest tests/integration_tests/agents/test_deep_agent_tools.py -v

# Run only Object Storage tests
pytest tests/integration_tests/agents/test_deep_agent_tools.py \
  -k "object_storage" -v

# Run only OpenSearch tests
pytest tests/integration_tests/agents/test_deep_agent_tools.py \
  -k "opensearch" -v

# Run only Oracle Vector tests
pytest tests/integration_tests/agents/test_deep_agent_tools.py \
  -k "oracle_vector" -v
```
"""

import os
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

# ============================================================================
# Skip Conditions
# ============================================================================


def skip_if_no_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    return os.environ.get("OCI_COMPARTMENT_ID") is None


def skip_if_no_deepagents() -> bool:
    """Check if deepagents is installed."""
    try:
        import deepagents  # noqa: F401

        return False
    except ImportError:
        return True


def skip_if_no_object_storage() -> bool:
    """Check if Object Storage config is available."""
    return (
        os.environ.get("OCI_NAMESPACE") is None or os.environ.get("OCI_BUCKET") is None
    )


def skip_if_no_opensearch() -> bool:
    """Check if OpenSearch config is available."""
    return os.environ.get("OPENSEARCH_HOST") is None


def skip_if_no_oracle_vector() -> bool:
    """Check if Oracle Vector Search config is available."""
    return (
        os.environ.get("ORACLE_DSN") is None
        or os.environ.get("ORACLE_USER") is None
        or os.environ.get("ORACLE_PASSWORD") is None
    )


# ============================================================================
# Object Storage Integration Tests
# ============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(skip_if_no_deepagents(), reason="deepagents not installed")
@pytest.mark.skipif(skip_if_no_oci_credentials(), reason="OCI credentials not set")
@pytest.mark.skipif(skip_if_no_object_storage(), reason="Object Storage not configured")
class TestDeepAgentWithObjectStorage:
    """Test Deep Agent with OCI Object Storage tools."""

    @pytest.fixture
    def object_storage_tools(self) -> list:
        """Create Object Storage tools."""
        from langchain_oci.tools import create_oci_object_storage_tools

        return create_oci_object_storage_tools(
            namespace=os.environ["OCI_NAMESPACE"],
            buckets=[os.environ["OCI_BUCKET"]],
            region=os.environ.get("OCI_REGION", "us-ashburn-1"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            enable_write=True,
            enable_delete=False,  # Safety: don't delete in tests
        )

    @pytest.fixture
    def agent_with_storage(self, object_storage_tools: list) -> Any:
        """Create deep agent with Object Storage tools."""
        from langchain_oci import create_deep_research_agent

        return create_deep_research_agent(
            tools=object_storage_tools,
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            system_prompt=(
                "You are a data analyst with access to OCI Object Storage. "
                "Use the tools to explore and analyze data in the bucket."
            ),
            temperature=0.3,
            max_tokens=2048,
        )

    def test_list_bucket_contents(self, agent_with_storage: Any) -> None:
        """Test agent can list bucket contents."""
        bucket = os.environ["OCI_BUCKET"]
        result = agent_with_storage.invoke(
            {
                "messages": [
                    HumanMessage(content=f"What files are in the {bucket} bucket?")
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content, "Should have a response"

    def test_read_and_analyze_file(self, agent_with_storage: Any) -> None:
        """Test agent can read and analyze files."""
        result = agent_with_storage.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "List the files in the bucket, then read any JSON file "
                            "you find and summarize its contents."
                        )
                    )
                ]
            }
        )

        assert "messages" in result
        # Should have used multiple tools
        message_types = [type(m).__name__ for m in result["messages"]]
        assert message_types.count("ToolMessage") >= 1

    def test_search_bucket_data(self, agent_with_storage: Any) -> None:
        """Test agent can search within bucket data."""
        bucket = os.environ["OCI_BUCKET"]
        result = agent_with_storage.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Search for 'research' in the {bucket} bucket data."
                    )
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content


# ============================================================================
# OpenSearch Integration Tests
# ============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents", "opensearch-py")
@pytest.mark.skipif(skip_if_no_deepagents(), reason="deepagents not installed")
@pytest.mark.skipif(skip_if_no_oci_credentials(), reason="OCI credentials not set")
@pytest.mark.skipif(skip_if_no_opensearch(), reason="OpenSearch not configured")
class TestDeepAgentWithOpenSearch:
    """Test Deep Agent with OCI OpenSearch tools."""

    @pytest.fixture
    def embedding_model(self) -> Any:
        """Create OCI embedding model for OpenSearch."""
        from langchain_oci import OCIGenAIEmbeddings

        return OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        )

    @pytest.fixture
    def opensearch_tools(self, embedding_model: Any) -> list:
        """Create OpenSearch tools."""
        from langchain_oci.tools import create_oci_opensearch_tools

        return create_oci_opensearch_tools(
            host=os.environ["OPENSEARCH_HOST"],
            index_name=os.environ.get("OPENSEARCH_INDEX", "research-vectors"),
            embedding_model=embedding_model,
            username=os.environ.get("OPENSEARCH_USER", "admin"),
            password=os.environ.get("OPENSEARCH_PASSWORD", ""),
            enable_write=False,  # Read-only for tests
            enable_delete=False,
        )

    @pytest.fixture
    def agent_with_opensearch(self, opensearch_tools: list) -> Any:
        """Create deep agent with OpenSearch tools."""
        from langchain_oci import create_deep_research_agent

        return create_deep_research_agent(
            tools=opensearch_tools,
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            system_prompt=(
                "You are a semantic search expert with access to a vector database. "
                "Use the search tools to find relevant information."
            ),
            temperature=0.3,
            max_tokens=2048,
        )

    def test_vector_search_query(self, agent_with_opensearch: Any) -> None:
        """Test agent can perform semantic vector search."""
        result = agent_with_opensearch.invoke(
            {
                "messages": [
                    HumanMessage(content="Search for documents about machine learning.")
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content

    def test_hybrid_search(self, agent_with_opensearch: Any) -> None:
        """Test agent can perform hybrid vector + keyword search."""
        result = agent_with_opensearch.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Do a hybrid search for 'neural network' combining "
                            "semantic meaning and keyword matching."
                        )
                    )
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content

    def test_get_index_stats(self, agent_with_opensearch: Any) -> None:
        """Test agent can retrieve index statistics."""
        result = agent_with_opensearch.invoke(
            {
                "messages": [
                    HumanMessage(content="How many documents are in the search index?")
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content


# ============================================================================
# Oracle Vector Search Integration Tests (26ai)
# ============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents", "oracledb")
@pytest.mark.skipif(skip_if_no_deepagents(), reason="deepagents not installed")
@pytest.mark.skipif(skip_if_no_oci_credentials(), reason="OCI credentials not set")
@pytest.mark.skipif(skip_if_no_oracle_vector(), reason="Oracle Vector not configured")
class TestDeepAgentWithOracleVector:
    """Test Deep Agent with Oracle 26ai Vector Search tools."""

    @pytest.fixture
    def embedding_model(self) -> Any:
        """Create OCI embedding model for Oracle Vector."""
        from langchain_oci import OCIGenAIEmbeddings

        return OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        )

    @pytest.fixture
    def oracle_vector_tools(self, embedding_model: Any) -> list:
        """Create Oracle Vector Search tools."""
        from langchain_oci.tools import create_oci_vector_search_tools

        return create_oci_vector_search_tools(
            dsn=os.environ["ORACLE_DSN"],
            user=os.environ["ORACLE_USER"],
            password=os.environ["ORACLE_PASSWORD"],
            wallet_location=os.environ.get("ORACLE_WALLET_LOCATION"),
            embedding_model=embedding_model,
            enable_write=False,  # Read-only for tests
            enable_delete=False,
            enable_hybrid=True,
        )

    @pytest.fixture
    def agent_with_oracle_vector(self, oracle_vector_tools: list) -> Any:
        """Create deep agent with Oracle Vector tools."""
        from langchain_oci import create_deep_research_agent

        return create_deep_research_agent(
            tools=oracle_vector_tools,
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            system_prompt=(
                "You are a knowledge base assistant with access to Oracle's "
                "vector database. Use semantic search to find relevant documents."
            ),
            temperature=0.3,
            max_tokens=2048,
        )

    def test_semantic_search(self, agent_with_oracle_vector: Any) -> None:
        """Test agent can perform semantic search in Oracle DB."""
        result = agent_with_oracle_vector.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Search for documents about cloud infrastructure."
                    )
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content

    def test_hybrid_search_with_keywords(self, agent_with_oracle_vector: Any) -> None:
        """Test agent can perform hybrid search with keyword filter."""
        result = agent_with_oracle_vector.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Find documents semantically related to 'data processing' "
                            "that also contain the keyword 'Oracle'."
                        )
                    )
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content

    def test_get_document_by_id(self, agent_with_oracle_vector: Any) -> None:
        """Test agent can retrieve specific documents."""
        result = agent_with_oracle_vector.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "First search for any document, then get the full "
                            "details of the first result by its ID."
                        )
                    )
                ]
            }
        )

        assert "messages" in result
        # Should have used multiple tools
        message_types = [type(m).__name__ for m in result["messages"]]
        assert message_types.count("ToolMessage") >= 1

    def test_database_statistics(self, agent_with_oracle_vector: Any) -> None:
        """Test agent can get database statistics."""
        result = agent_with_oracle_vector.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="How many documents are in the vector database?"
                    )
                ]
            }
        )

        assert "messages" in result
        final = result["messages"][-1]
        assert final.content


# ============================================================================
# Combined Tools Integration Test
# ============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(skip_if_no_deepagents(), reason="deepagents not installed")
@pytest.mark.skipif(skip_if_no_oci_credentials(), reason="OCI credentials not set")
@pytest.mark.skipif(skip_if_no_object_storage(), reason="Object Storage not configured")
class TestDeepAgentMultipleToolSets:
    """Test Deep Agent with multiple OCI tool sets combined."""

    @pytest.fixture
    def combined_tools(self) -> list:
        """Create combined Object Storage tools and mock research tools."""
        from langchain_core.tools import tool

        from langchain_oci.tools import create_oci_object_storage_tools

        # Real Object Storage tools
        storage_tools = create_oci_object_storage_tools(
            namespace=os.environ["OCI_NAMESPACE"],
            buckets=[os.environ["OCI_BUCKET"]],
            region=os.environ.get("OCI_REGION", "us-ashburn-1"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            enable_write=False,
        )

        # Additional analysis tool
        @tool
        def analyze_data(content: str) -> str:
            """Analyze data and provide insights."""
            word_count = len(content.split())
            char_count = len(content)
            return (
                f"Analysis: {word_count} words, {char_count} characters. "
                "Data appears to be structured content suitable for processing."
            )

        return storage_tools + [analyze_data]

    @pytest.fixture
    def agent_with_combined_tools(self, combined_tools: list) -> Any:
        """Create deep agent with combined tools."""
        from langchain_oci import create_deep_research_agent

        return create_deep_research_agent(
            tools=combined_tools,
            compartment_id=os.environ["OCI_COMPARTMENT_ID"],
            service_endpoint=os.environ.get(
                "OCI_SERVICE_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
            system_prompt=(
                "You are a research analyst with access to cloud storage and "
                "analysis tools. Combine data retrieval with analysis."
            ),
            temperature=0.3,
            max_tokens=2048,
        )

    def test_fetch_and_analyze_workflow(self, agent_with_combined_tools: Any) -> None:
        """Test agent can fetch data from storage and analyze it."""
        result = agent_with_combined_tools.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "List the files in the bucket, read any file you find, "
                            "then analyze its content."
                        )
                    )
                ]
            }
        )

        assert "messages" in result
        # Should have used multiple tools
        message_types = [type(m).__name__ for m in result["messages"]]
        tool_calls = message_types.count("ToolMessage")
        assert tool_calls >= 2, f"Expected at least 2 tool calls, got {tool_calls}"
