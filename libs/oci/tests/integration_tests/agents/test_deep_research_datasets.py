# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests using public datasets (Medical & Legal).

These tests use real datasets from Hugging Face to evaluate
the deep research agent on domain-specific tasks.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install langchain-oci deepagents datasets
   ```

2. Set up OCI credentials:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment..."
   export OCI_REGION="us-chicago-1"
   ```

## Running Tests

```bash
# Run all dataset tests
pytest tests/integration_tests/agents/test_deep_agent_datasets.py -v

# Run only medical tests
pytest tests/integration_tests/agents/test_deep_agent_datasets.py -k medical

# Run only legal tests
pytest tests/integration_tests/agents/test_deep_agent_datasets.py -k legal
```
"""

import os
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# =============================================================================
# MOCK TOOLS FOR TESTING
# =============================================================================


@tool
def search_medical_knowledge(query: str) -> str:
    """Search medical knowledge base for information."""
    # Mock medical knowledge
    knowledge = {
        "pathology": (
            "Pathology is the study of disease causes and effects. "
            "Key areas: anatomical pathology, clinical pathology, molecular "
            "pathology. Common findings: inflammation, necrosis, neoplasia."
        ),
        "leukocytosis": (
            "Leukocytosis is elevated white blood cell count (>11,000/mcL). "
            "Causes: infection, inflammation, leukemia. Leukemoid reaction: "
            "extreme leukocytosis (>50,000/mcL) mimicking leukemia."
        ),
        "infection": (
            "Bacterial infections cause inflammatory response. "
            "Pseudomonas aeruginosa: gram-negative, opportunistic pathogen. "
            "Commonly causes pneumonia, UTIs, wound infections."
        ),
    }
    for key, info in knowledge.items():
        if key in query.lower():
            return f"Medical Knowledge:\n{info}"
    return f"General medical information for: {query}"


@tool
def search_legal_database(query: str) -> str:
    """Search legal database for contract and regulatory information."""
    # Mock legal knowledge
    knowledge = {
        "termination": (
            "Contract termination clauses define how parties can end agreement. "
            "Types: termination for cause, termination for convenience. "
            "Notice periods typically 30-90 days."
        ),
        "indemnification": (
            "Indemnification clauses allocate risk between parties. "
            "Indemnitor agrees to compensate indemnitee for losses. "
            "Often covers third-party claims, IP infringement."
        ),
        "liability": (
            "Liability caps limit maximum damages recoverable. "
            "Common caps: contract value, 12 months fees. "
            "Carve-outs: IP infringement, gross negligence, data breach."
        ),
    }
    for key, info in knowledge.items():
        if key in query.lower():
            return f"Legal Database:\n{info}"
    return f"General legal information for: {query}"


@tool
def analyze_document(text: str) -> str:
    """Analyze a document and extract key information."""
    word_count = len(text.split())
    return (
        f"Document Analysis:\n"
        f"- Word count: {word_count}\n"
        f"- Contains key terms identified\n"
        f"- Structure: formal document format"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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


def skip_if_no_datasets() -> bool:
    """Check if datasets library is installed."""
    try:
        import datasets  # noqa: F401

        return False
    except ImportError:
        return True


def create_test_agent(tools: list[Any]) -> Any:
    """Create a configured deep agent for testing."""
    from langchain_oci import create_deep_research_agent

    compartment_id = os.environ.get("OCI_COMPARTMENT_ID", "")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    return create_deep_research_agent(
        tools=tools,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        system_prompt=(
            "You are an expert research analyst. Use available tools to "
            "research and provide detailed, accurate answers. Always cite "
            "your reasoning and provide clear explanations."
        ),
        temperature=0.3,
        max_tokens=2048,
    )


# =============================================================================
# MEDICAL DATASET TESTS
# =============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents", "datasets")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.skipif(
    skip_if_no_datasets(),
    reason="datasets package not installed",
)
class TestMedicalDatasets:
    """Tests using MedMCQA medical question dataset."""

    def test_medical_qa_pathology(self) -> None:
        """Test agent on pathology questions from MedMCQA."""
        from tests.integration_tests.agents.public_datasets import (
            load_medical_qa_samples,
        )

        # Load pathology questions
        tasks = load_medical_qa_samples(n_samples=2, subject="Pathology")
        if not tasks:
            pytest.skip("Could not load medical dataset")

        agent = create_test_agent([search_medical_knowledge, analyze_document])

        for task in tasks:
            result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})

            # Verify response
            assert "messages" in result
            final_message = result["messages"][-1]
            assert final_message.content, f"No response for task {task['id']}"
            assert len(final_message.content) > 50, "Response too short"

    def test_medical_qa_multiple_subjects(self) -> None:
        """Test agent on questions from multiple medical subjects."""
        from tests.integration_tests.agents.public_datasets import (
            load_medical_qa_samples,
        )

        # Load general medical questions
        tasks = load_medical_qa_samples(n_samples=3)
        if not tasks:
            pytest.skip("Could not load medical dataset")

        agent = create_test_agent([search_medical_knowledge])

        results = []
        for task in tasks:
            result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})
            final_message = result["messages"][-1]
            results.append(
                {
                    "task_id": task["id"],
                    "subject": task["subject"],
                    "has_response": bool(final_message.content),
                    "response_length": len(final_message.content),
                }
            )

        # All tasks should have responses
        assert all(r["has_response"] for r in results)


@pytest.mark.requires("oci", "langgraph", "deepagents", "datasets")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.skipif(
    skip_if_no_datasets(),
    reason="datasets package not installed",
)
class TestPubMedQADatasets:
    """Tests using PubMedQA research question dataset."""

    def test_pubmed_research_questions(self) -> None:
        """Test agent on biomedical research questions."""
        from tests.integration_tests.agents.public_datasets import (
            load_pubmed_qa_samples,
        )

        tasks = load_pubmed_qa_samples(n_samples=2)
        if not tasks:
            pytest.skip("Could not load PubMedQA dataset")

        agent = create_test_agent([search_medical_knowledge, analyze_document])

        for task in tasks:
            # Truncate query if too long
            query = task["query"][:3000] if len(task["query"]) > 3000 else task["query"]

            result = agent.invoke({"messages": [HumanMessage(content=query)]})

            final_message = result["messages"][-1]
            assert final_message.content, f"No response for {task['id']}"


# =============================================================================
# LEGAL DATASET TESTS
# =============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents", "datasets")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.skipif(
    skip_if_no_datasets(),
    reason="datasets package not installed",
)
class TestLegalDatasets:
    """Tests using CUAD legal contract dataset."""

    def test_legal_contract_analysis(self) -> None:
        """Test agent on contract clause analysis."""
        from tests.integration_tests.agents.public_datasets import (
            load_legal_qa_samples,
        )

        tasks = load_legal_qa_samples(n_samples=2)
        if not tasks:
            pytest.skip("Could not load legal dataset")

        agent = create_test_agent([search_legal_database, analyze_document])

        for task in tasks:
            result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})

            final_message = result["messages"][-1]
            assert final_message.content, f"No response for {task['id']}"
            # Legal analysis should be substantive
            assert len(final_message.content) > 100, "Legal analysis too brief"


# =============================================================================
# BENCHMARK TESTS
# =============================================================================


@pytest.mark.requires("oci", "langgraph", "deepagents", "datasets")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.skipif(
    skip_if_no_datasets(),
    reason="datasets package not installed",
)
def test_cross_domain_benchmark() -> None:
    """Test agent on benchmark tasks from multiple domains."""
    from tests.integration_tests.agents.public_datasets import (
        load_research_benchmark,
    )

    # Load 2 samples from each domain
    benchmark = load_research_benchmark(
        domains=["medical", "legal"],
        samples_per_domain=2,
    )

    if not benchmark:
        pytest.skip("Could not load benchmark datasets")

    agent = create_test_agent(
        [
            search_medical_knowledge,
            search_legal_database,
            analyze_document,
        ]
    )

    results = []
    for task in benchmark:
        # Truncate long queries
        query = task["query"][:2500]

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        final_message = result["messages"][-1]
        results.append(
            {
                "id": task["id"],
                "domain": task["domain"],
                "source": task["source"],
                "success": bool(final_message.content),
                "length": len(final_message.content) if final_message.content else 0,
            }
        )

    # Report results
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below threshold"

    # Check domain coverage
    domains_tested = set(r["domain"] for r in results)
    assert len(domains_tested) >= 2, "Should test multiple domains"
