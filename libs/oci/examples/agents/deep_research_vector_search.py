#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Deep Research Agent with OCI Vector Search.

This example demonstrates using the deep research agent with Oracle 26ai
vector search for semantic document retrieval across multiple domains.

## 1) Data source (where ADB data comes from)

This example queries ADB table `VECTOR_DOCUMENTS`. In this repository, data is
loaded through:
1. `scripts/upload_research_datasets.py` (Hugging Face -> OCI Object Storage)
2. `scripts/upload_large_datasets.py` (optional larger corpora)
3. `scripts/vectorize_datasets.py` (Object Storage -> ADB with embeddings)

Typical dataset origins in this flow:
- MedMCQA, PubMedQA, CUAD from Hugging Face (research/QA and legal clauses)
- optional larger corpora like Wikipedia, C4, ArXiv

## 2) Embedding model used

This example explicitly uses:
`cohere.embed-v4.0`.

This is the same model used by default in `langchain-oci` datastore tooling.

## 3) Search implementation used here

This file intentionally implements a custom tool class:
- `VectorSearchTool` (local class in this file)

It does not use the built-in `ADB` datastore adapter directly. For the canonical
SDK integration path (`ADB` + `create_deep_research_agent(datastores=...)`), see:
- `examples/agents/deep_research_adb_datastore.py`

## 4) What you provide at runtime

- ADB connection details (`DB_DSN`, `DB_USER`, `DB_PASSWORD`)
- OCI config for embeddings and chat model (`OCI_COMPARTMENT_ID`, `OCI_REGION`)
- user research prompt/query

This script itself does not ingest data; it performs query-time semantic search
over documents that are already indexed in ADB.

## Quick answers (review checklist)

- Data source: ADB `VECTOR_DOCUMENTS`, pre-populated via ingestion scripts.
- Embeddings: explicitly passed as `cohere.embed-v4.0` in this file.
- Additional ADB search class: yes, intentionally in this gist (`VectorSearchTool`)
  to show custom-tool pattern.
- Runtime inputs: ADB connection, OCI config, and user prompt.

## Prerequisites

1. Set up OCI credentials
2. Ensure the vector database is populated (run scripts/vectorize_datasets.py)

## Running

```bash
python examples/agents/deep_research_vector_search.py
```
"""

import os
from typing import Any, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import Field

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)
GENAI_REGION = os.environ.get("OCI_REGION", "us-chicago-1")

# Database configuration (TLS connection, no mTLS)
DB_DSN = "tcps://adb.ca-toronto-1.oraclecloud.com:1522/g4549e8afff78c6_deepresearch_low.adb.oraclecloud.com"
DB_USER = "ADMIN"
DB_PASSWORD = "Research2026Pass#"


class VectorSearchTool(BaseTool):
    """Semantic search tool using Oracle 26ai vector database."""

    name: str = "semantic_search"
    description: str = (
        "Search for documents using semantic similarity. "
        "This finds conceptually related content, not just keyword matches. "
        "Available datasets: wikipedia, c4 (web), pubmedqa (medical), "
        "medmcqa (medical), cuad (legal contracts). "
        "Input: A natural language query. "
        "Optional: Specify dataset filter like 'medical: symptoms of diabetes' "
        "or 'legal: termination clause'."
    )

    connection: Any = Field(default=None, exclude=True)
    embedding_model: Any = Field(default=None, exclude=True)
    top_k: int = 5

    def _run(self, query: str) -> str:
        """Search for semantically similar documents."""
        import json

        # Parse optional dataset filter
        dataset_filter: Optional[Tuple[str, ...]] = None
        if ":" in query:
            prefix, rest = query.split(":", 1)
            prefix = prefix.strip().lower()
            if prefix in ["medical", "legal", "wikipedia", "web", "c4"]:
                query = rest.strip()
                if prefix == "medical":
                    dataset_filter = ("pubmedqa", "medmcqa")
                elif prefix == "legal":
                    dataset_filter = ("cuad",)
                elif prefix in ("web", "c4"):
                    dataset_filter = ("c4",)
                else:
                    dataset_filter = (prefix,)

        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.embed_query(query)

            cursor = self.connection.cursor()

            # Build SQL with optional dataset filter
            if dataset_filter:
                placeholders = ",".join([f":ds{i}" for i in range(len(dataset_filter))])
                sql = f"""
                    SELECT id, title, content, source, dataset,
                           VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
                    FROM VECTOR_DOCUMENTS
                    WHERE dataset IN ({placeholders})
                    ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                    FETCH FIRST :top_k ROWS ONLY
                """
                params = {
                    "query_vec": json.dumps(query_embedding),
                    "top_k": self.top_k,
                }
                for i, ds in enumerate(dataset_filter):
                    params[f"ds{i}"] = ds
            else:
                sql = """
                    SELECT id, title, content, source, dataset,
                           VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
                    FROM VECTOR_DOCUMENTS
                    ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                    FETCH FIRST :top_k ROWS ONLY
                """
                params = {
                    "query_vec": json.dumps(query_embedding),
                    "top_k": self.top_k,
                }

            cursor.execute(sql, params)

            results = []
            for row in cursor:
                results.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "content": row[2][:1500],  # Truncate for display
                        "source": row[3],
                        "dataset": row[4],
                        "similarity": 1 - row[5],  # Convert distance to similarity
                    }
                )

            cursor.close()

            if not results:
                return f"No results found for query: {query}"

            output = f"Found {len(results)} relevant documents:\n\n"
            for i, doc in enumerate(results, 1):
                sim = doc["similarity"]
                ds = doc["dataset"]
                output += f"--- Result {i} (similarity: {sim:.3f}, dataset: {ds}) ---\n"
                output += f"Title: {doc['title']}\n"
                output += f"Source: {doc['source']}\n"
                output += f"Content: {doc['content']}...\n\n"

            return output

        except Exception as e:
            return f"Error performing search: {e}"


def create_vector_search_tool() -> VectorSearchTool:
    """Create the vector search tool with database connection."""
    import oracledb

    from langchain_oci import OCIGenAIEmbeddings

    # Create embedding model
    embedding_model = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "API_KEY_AUTH"),
    )

    # Create database connection
    connection = oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DB_DSN,
    )

    return VectorSearchTool(
        connection=connection,
        embedding_model=embedding_model,
        top_k=5,
    )


def main():
    """Run deep research agent with vector search."""
    from langchain_oci.agents import create_deep_research_agent

    print("=" * 60)
    print("Deep Research Agent with Oracle 26ai Vector Search")
    print("=" * 60)

    # Create vector search tool
    print("\nInitializing vector search tool...")
    search_tool = create_vector_search_tool()
    print("  Connected to Oracle 26ai vector database")

    # Create deep research agent
    print("\nCreating deep research agent...")
    service_endpoint = (
        f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com"
    )

    agent = create_deep_research_agent(
        tools=[search_tool],
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "API_KEY_AUTH"),
        system_prompt=(
            "You are a deep research analyst with access to a semantic search tool "
            "that can find relevant documents across multiple domains:\n"
            "- Wikipedia articles (general knowledge)\n"
            "- C4 web corpus (diverse web content)\n"
            "- PubMedQA (biomedical research Q&A)\n"
            "- MedMCQA (medical exam questions)\n"
            "- CUAD (legal contract clauses)\n\n"
            "Use the semantic_search tool to find relevant information. "
            "You can filter by domain using prefixes like 'medical:', 'legal:', "
            "'wikipedia:', or 'web:' followed by your query.\n\n"
            "Always cite your sources and synthesize information from multiple results."
        ),
        temperature=0.3,
        max_tokens=2048,
    )
    print("  Agent created successfully!")

    # Example research queries
    queries = [
        "What are the symptoms and treatments for leukocytosis?",
        "legal: What termination clauses are common in contracts?",
        "wikipedia: Explain the history and significance of the Turing test",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Research Query {i}:")
        print(f"{'=' * 60}")
        print(f"\n{query}\n")

        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})

            # Extract response
            final_message = result["messages"][-1]
            print("\n--- Agent Response ---")
            print(final_message.content)

            # Show tool usage
            tool_messages = [
                m for m in result["messages"] if type(m).__name__ == "ToolMessage"
            ]
            if tool_messages:
                print(f"\n(Used {len(tool_messages)} tool calls)")

        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
