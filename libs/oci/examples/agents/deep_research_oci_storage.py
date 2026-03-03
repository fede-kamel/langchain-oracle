#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201, I001

"""Deep Research Agent with OCI Object Storage.

This example demonstrates using the deep research agent with OCI Object Storage
tools to analyze medical and legal datasets stored in OCI buckets.

## 1) Data source (where data comes from)

This example reads JSON datasets from OCI Object Storage buckets.

In this repository, those datasets are created by:
1. `scripts/upload_research_datasets.py`:
   - downloads MedMCQA, PubMedQA, and CUAD from Hugging Face
   - uploads them to `deep-research-medical` and `deep-research-legal`
2. `scripts/upload_large_datasets.py` (optional):
   - uploads large corpora (Wikipedia, C4, ArXiv) to `deep-research-large`

## 2) Embedding model used

This example does not do vector retrieval, so it does not need query embeddings.
The LLM model is used for reasoning/synthesis over retrieved object data.

## 3) Search implementation used here

This file uses Object Storage tools (`list/read/search` over bucket objects),
not ADB/OpenSearch vector search.

## 4) What you provide at runtime

- bucket/namespace configuration and OCI credentials
- the research prompt/query

This script does not write embeddings or populate ADB.

## Quick answers (review checklist)

- Data source: OCI Object Storage buckets filled by repository upload scripts.
- Embeddings: not applicable in this gist (no vector datastore retrieval path).
- Additional ADB search class: no; this gist uses Object Storage tools only.
- Runtime inputs: OCI storage config and user prompt.

## Prerequisites

1. Set up OCI credentials:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment..."
   export OCI_REGION="us-chicago-1"  # For GenAI
   ```

2. Install dependencies:
   ```bash
   pip install langchain-oci deepagents
   ```

3. Ensure datasets are uploaded to OCI Object Storage:
   ```bash
   python scripts/upload_research_datasets.py
   ```

## Running

```bash
python examples/agents/deep_research_oci_storage.py
```
"""

import os

from langchain_core.messages import HumanMessage

from langchain_oci.agents import create_deep_research_agent
from langchain_oci.tools import create_oci_object_storage_tools  # type: ignore[attr-defined]

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaaih6r7yauu3rloxnf4gj7uidfr5sbtsanxgnej5zfxu27gsbn6o5a",
)
GENAI_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
STORAGE_REGION = "us-ashburn-1"
NAMESPACE = "id0qhv5yj7ke"
AUTH_PROFILE = "API_KEY_AUTH"

# Bucket names
MEDICAL_BUCKET = "deep-research-medical"
LEGAL_BUCKET = "deep-research-legal"
LARGE_BUCKET = "deep-research-large"  # Wikipedia, C4, ArXiv


def main():
    """Run deep research agent with OCI Object Storage."""
    print("=" * 60)
    print("Deep Research Agent with OCI Object Storage")
    print("=" * 60)

    # Create OCI Object Storage tools
    print("\nCreating OCI Object Storage tools...")
    storage_tools = create_oci_object_storage_tools(
        namespace=NAMESPACE,
        buckets=[MEDICAL_BUCKET, LEGAL_BUCKET, LARGE_BUCKET],
        region=STORAGE_REGION,
        auth_profile=AUTH_PROFILE,
    )
    print(f"  Created {len(storage_tools)} tools:")
    for tool in storage_tools:
        print(f"    - {tool.name}: {tool.description[:60]}...")

    # Create deep research agent
    print("\nCreating deep research agent...")
    service_endpoint = (
        f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com"
    )

    agent = create_deep_research_agent(
        tools=storage_tools,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=AUTH_PROFILE,
        system_prompt=(
            "You are a deep research analyst with access to massive datasets "
            "stored in OCI Object Storage. "
            "Available buckets:\n"
            f"- {MEDICAL_BUCKET}: MedMCQA (medical questions) and "
            "PubMedQA (biomedical research)\n"
            f"- {LEGAL_BUCKET}: CUAD (contract clauses)\n"
            f"- {LARGE_BUCKET}: Wikipedia (1M+ articles), C4 web corpus (1M+), "
            "ArXiv papers (100K+)\n\n"
            "Use the tools to:\n"
            "1. list_bucket_objects: Discover available data files\n"
            "2. read_bucket_object: Read specific JSON files\n"
            "3. search_bucket_data: Search for specific terms\n\n"
            "Always cite your sources and provide thorough analysis."
        ),
        temperature=0.3,
        max_tokens=2048,
    )
    print("  Agent created successfully!")

    # Example research queries
    queries = [
        "What pathology questions are available in the medical dataset? "
        "List a few examples and explain the topics they cover.",
        "Search the medical bucket for questions about 'leukocytosis'. "
        "What information is available?",
        "What types of contract clauses are in the legal dataset? "
        "Give examples of termination or indemnification clauses.",
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
