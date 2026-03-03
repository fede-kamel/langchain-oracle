# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for datastore tools with react and deep_research agents.

These tests verify that datastore tools work with both agent types.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication
2. **Environment Variables**:
   ```bash
   export OCI_REGION="us-chicago-1"
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   ```

## Running the Tests

```bash
cd libs/oci
python -m pytest tests/integration_tests/agents/test_datastore_tools_integration.py -v
```
"""

import os
from typing import Any, Optional

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.agents.datastores.vectorstores import VectorDataStore


class MockVectorStore(VectorDataStore):
    """Mock vector store for testing without real database connections."""

    def __init__(self, name_: str = "mock", hint_: str = "", documents: list = None):
        self._name = name_
        self._hint = hint_
        self._documents = documents or []
        self._embedding_model = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hint(self) -> str:
        return self._hint

    def connect(self, embedding_model: Any) -> None:
        self._embedding_model = embedding_model

    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        # Simple mock: return documents that contain query terms
        results = []
        for doc in self._documents:
            content = doc.get("content", "").lower()
            if any(term in content for term in query.lower().split()):
                results.append({**doc, "score": 0.9})
        return results[:top_k]

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        results = []
        for doc in self._documents:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()
            if query.lower() in content or query.lower() in title:
                results.append(doc)
        return results[:top_k]

    def get(self, document_id: str | int) -> Optional[dict]:
        for doc in self._documents:
            if str(doc.get("id")) == str(document_id):
                return doc
        return None

    def insert(
        self, title: str, content: str, source: str, embedding: list[float]
    ) -> str:
        doc_id = str(len(self._documents) + 1)
        self._documents.append(
            {
                "id": doc_id,
                "title": title,
                "content": content,
                "source": source,
            }
        )
        return doc_id

    def bulk_insert(self, documents: list[dict], embeddings: list[list[float]]) -> int:
        for doc in documents:
            self.insert(
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("source", ""),
                [],
            )
        return len(documents)

    def update(
        self,
        document_id: str | int,
        title: Optional[str],
        content: Optional[str],
        source: Optional[str],
        embedding: Optional[list[float]],
    ) -> bool:
        for doc in self._documents:
            if str(doc.get("id")) == str(document_id):
                if title:
                    doc["title"] = title
                if content:
                    doc["content"] = content
                if source:
                    doc["source"] = source
                return True
        return False

    def delete(self, document_id: str | int) -> bool:
        for i, doc in enumerate(self._documents):
            if str(doc.get("id")) == str(document_id):
                self._documents.pop(i)
                return True
        return False

    def stats(self) -> dict:
        return {
            "store": self._name,
            "document_count": len(self._documents),
        }


# Sample documents for testing
HR_DOCUMENTS = [
    {
        "id": "1",
        "title": "PTO Policy",
        "content": (
            "Employees receive 20 days of PTO per year. "
            "Unused PTO can be carried over up to 5 days."
        ),
        "source": "hr_handbook",
    },
    {
        "id": "2",
        "title": "Benefits Overview",
        "content": (
            "Health insurance, dental, vision, and 401k matching up to 6% "
            "are included in the benefits package."
        ),
        "source": "hr_handbook",
    },
    {
        "id": "3",
        "title": "Remote Work Policy",
        "content": (
            "Employees may work remotely up to 3 days per week with manager approval."
        ),
        "source": "hr_handbook",
    },
]

SALES_DOCUMENTS = [
    {
        "id": "1",
        "title": "Q4 2025 Sales Report",
        "content": (
            "Q4 revenue reached $50M, a 15% increase over Q3. "
            "Top products: Cloud Services (40%), Software Licenses (35%), "
            "Support (25%)."
        ),
        "source": "sales_reports",
    },
    {
        "id": "2",
        "title": "Customer Analysis",
        "content": (
            "Enterprise customers account for 70% of revenue. "
            "Customer retention rate: 92%. Net Promoter Score: 45."
        ),
        "source": "sales_analytics",
    },
]


def skip_if_no_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    return os.environ.get("OCI_COMPARTMENT_ID") is None


@pytest.mark.requires("oci", "langgraph")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
class TestDatastoreToolsWithReactAgent:
    """Test datastore tools with create_oci_agent (ReAct)."""

    @pytest.fixture
    def mock_stores(self) -> dict[str, MockVectorStore]:
        """Create mock stores with sample data."""
        return {
            "hr": MockVectorStore(
                name_="hr",
                hint_="HR policies, PTO, benefits, remote work",
                documents=HR_DOCUMENTS.copy(),
            ),
            "sales": MockVectorStore(
                name_="sales",
                hint_="Sales data, revenue, customers, Q4 reports",
                documents=SALES_DOCUMENTS.copy(),
            ),
        }

    @pytest.fixture
    def compartment_id(self) -> str:
        return os.environ.get("OCI_COMPARTMENT_ID", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    def test_agent_with_datastore_tools(
        self,
        mock_stores: dict,
        compartment_id: str,
        service_endpoint: str,
    ) -> None:
        """Test that react agent can use datastore tools."""
        from langchain_oci import create_datastore_tools, create_oci_agent

        # Create tools with mock stores
        tools = create_datastore_tools(
            stores=mock_stores,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
        )

        agent = create_oci_agent(
            model_id="meta.llama-3.3-70b-instruct",
            tools=tools,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            system_prompt=(
                "You are an assistant with access to HR and Sales databases. "
                "Use the search tools to find information."
            ),
            temperature=0.3,
            max_tokens=1024,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="What is the PTO policy?")]}
        )

        assert "messages" in result
        assert len(result["messages"]) > 1

    def test_stats_tool(
        self,
        mock_stores: dict,
        compartment_id: str,
        service_endpoint: str,
    ) -> None:
        """Test that stats tool returns store statistics."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=mock_stores,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
        )

        stats_tool = next(t for t in tools if t.name == "stats")
        result = stats_tool._run(store=None)

        assert "hr" in result
        assert "sales" in result
        assert "Documents" in result


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
class TestDatastoreToolsWithDeepResearchAgent:
    """Test datastore tools with create_deep_research_agent."""

    def skip_if_no_deepagents(self) -> bool:
        try:
            import deepagents  # noqa: F401

            return False
        except ImportError:
            return True

    @pytest.fixture
    def mock_stores(self) -> dict[str, MockVectorStore]:
        return {
            "docs": MockVectorStore(
                name_="docs",
                hint_="Company documentation",
                documents=HR_DOCUMENTS.copy() + SALES_DOCUMENTS.copy(),
            ),
        }

    @pytest.fixture
    def compartment_id(self) -> str:
        return os.environ.get("OCI_COMPARTMENT_ID", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    def test_deep_agent_with_datastores(
        self,
        mock_stores: dict,
        compartment_id: str,
        service_endpoint: str,
    ) -> None:
        """Test that deep research agent works with datastores parameter."""
        try:
            import deepagents  # noqa: F401
        except ImportError:
            pytest.skip("deepagents not installed")

        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            datastores=mock_stores,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            model_id="google.gemini-2.5-pro",  # Use Gemini 2.5 Pro
            temperature=0.3,
            max_tokens=2048,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="What are the company benefits?")]}
        )

        assert "messages" in result
        assert len(result["messages"]) > 1


@pytest.mark.requires("oci")
class TestMockVectorStore:
    """Tests for MockVectorStore implementation (for test validation)."""

    def test_search(self) -> None:
        """Test mock store search functionality."""
        store = MockVectorStore(
            name_="test",
            documents=[
                {"id": "1", "title": "Test", "content": "Hello world"},
                {"id": "2", "title": "Other", "content": "Goodbye moon"},
            ],
        )
        store.connect(None)

        results = store.search("hello", [], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "1"

    def test_keyword_search(self) -> None:
        """Test mock store keyword search."""
        store = MockVectorStore(
            name_="test",
            documents=[
                {"id": "1", "title": "Test", "content": "Hello world"},
            ],
        )

        results = store.keyword_search("hello", top_k=5)
        assert len(results) == 1

    def test_get(self) -> None:
        """Test mock store get by ID."""
        store = MockVectorStore(
            name_="test",
            documents=[{"id": "42", "title": "Found", "content": "Content"}],
        )

        doc = store.get("42")
        assert doc is not None
        assert doc["title"] == "Found"

        missing = store.get("999")
        assert missing is None

    def test_stats(self) -> None:
        """Test mock store stats."""
        store = MockVectorStore(
            name_="test",
            documents=[{"id": "1"}, {"id": "2"}],
        )

        stats = store.stats()
        assert stats["document_count"] == 2
        assert stats["store"] == "test"
