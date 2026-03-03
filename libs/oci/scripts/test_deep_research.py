#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Test the Deep Research Agent with Oracle 26ai Vector Search.

This script tests:
1. Database connectivity
2. Vector search across different domains
3. Deep research agent with semantic search tool
"""

import json
import os

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)
GENAI_REGION = "us-chicago-1"

# Database
DB_DSN = "tcps://adb.ca-toronto-1.oraclecloud.com:1522/g4549e8afff78c6_deepresearch_low.adb.oraclecloud.com"
DB_USER = "ADMIN"
DB_PASSWORD = "Research2026Pass#"


def test_database_connection():
    """Test 1: Database connectivity."""
    print("\n" + "=" * 60)
    print("TEST 1: Database Connectivity")
    print("=" * 60)

    import oracledb

    try:
        conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)
        cursor = conn.cursor()

        # Get version
        cursor.execute("SELECT BANNER FROM V$VERSION")
        version = cursor.fetchone()[0]
        print(f"  Connected to: {version}")

        # Get document count
        cursor.execute("SELECT COUNT(*) FROM VECTOR_DOCUMENTS")
        count = cursor.fetchone()[0]
        print(f"  Total documents: {count:,}")

        # Get counts by dataset
        cursor.execute("""
            SELECT dataset, COUNT(*) as cnt
            FROM VECTOR_DOCUMENTS
            GROUP BY dataset
            ORDER BY cnt DESC
        """)
        print("  By dataset:")
        for row in cursor:
            print(f"    - {row[0]}: {row[1]:,}")

        cursor.close()
        conn.close()
        print("\n  [PASS] Database connection successful")
        return True
    except Exception as e:
        print(f"\n  [FAIL] Database error: {e}")
        return False


def test_vector_search():
    """Test 2: Vector search across domains."""
    print("\n" + "=" * 60)
    print("TEST 2: Vector Search Across Domains")
    print("=" * 60)

    import oci
    import oracledb
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    from oci.generative_ai_inference.models import (
        EmbedTextDetails,
        OnDemandServingMode,
    )

    # Setup
    config = oci.config.from_file(profile_name="API_KEY_AUTH")
    config["region"] = GENAI_REGION

    genai_client = GenerativeAiInferenceClient(
        config,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
    )

    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)
    cursor = conn.cursor()

    # Test queries for different domains
    test_queries = [
        ("medical", "symptoms and treatment of hypertension"),
        ("legal", "contract termination clause"),
        ("wikipedia", "artificial intelligence history"),
        ("general", "climate change effects"),
    ]

    all_passed = True
    for domain, query in test_queries:
        print(f"\n  Query ({domain}): '{query}'")

        try:
            # Generate embedding
            embed_request = EmbedTextDetails(
                inputs=[query],
                serving_mode=OnDemandServingMode(model_id="cohere.embed-english-v3.0"),
                compartment_id=COMPARTMENT_ID,
                input_type="SEARCH_QUERY",
                truncate="END",
            )
            response = genai_client.embed_text(embed_request)
            query_embedding = response.data.embeddings[0]

            # Search
            sql = """
                SELECT title, dataset,
                       VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
                FROM VECTOR_DOCUMENTS
                ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                FETCH FIRST 3 ROWS ONLY
            """
            cursor.execute(sql, {"query_vec": json.dumps(query_embedding)})

            print("  Results:")
            for i, row in enumerate(cursor, 1):
                similarity = 1 - row[2]
                title = row[0][:50] + "..." if len(row[0]) > 50 else row[0]
                print(f"    {i}. [{row[1]}] {title} (sim: {similarity:.3f})")

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            all_passed = False

    cursor.close()
    conn.close()

    if all_passed:
        print("\n  [PASS] Vector search working across all domains")
    return all_passed


def test_deep_research_agent():
    """Test 3: Research Agent with semantic search."""
    print("\n" + "=" * 60)
    print("TEST 3: Research Agent with Vector Search")
    print("=" * 60)

    import oracledb
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from langchain_oci import OCIGenAIEmbeddings
    from langchain_oci.agents import create_oci_agent  # Use simpler ReAct agent

    # Setup connections first (used in tool closure)
    print("\n  Setting up connections...")
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)

    embedding_model = OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
    )

    # Create tool using decorator (closure captures conn and embedding_model)
    @tool
    def search(query: str) -> str:
        """Search for relevant documents using semantic similarity.

        Use prefixes to filter by domain:
        - 'medical:' for medical/health content
        - 'legal:' for legal/contract content
        - 'wikipedia:' for encyclopedia articles
        - 'web:' for general web content

        Args:
            query: The search query, optionally with a domain prefix
        """
        # Parse domain filter
        dataset_filter = None
        search_query = query
        if ":" in query:
            prefix, rest = query.split(":", 1)
            prefix = prefix.strip().lower()
            if prefix in ["medical", "legal", "wikipedia", "web"]:
                search_query = rest.strip()
                if prefix == "medical":
                    dataset_filter = ("pubmedqa", "medmcqa")
                elif prefix == "legal":
                    dataset_filter = ("cuad",)
                elif prefix == "web":
                    dataset_filter = ("c4",)
                else:
                    dataset_filter = (prefix,)

        # Generate embedding
        query_embedding = embedding_model.embed_query(search_query)

        cursor = conn.cursor()

        if dataset_filter:
            placeholders = ",".join([f":ds{i}" for i in range(len(dataset_filter))])
            sql = f"""
                SELECT title, content, dataset
                FROM VECTOR_DOCUMENTS
                WHERE dataset IN ({placeholders})
                ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                FETCH FIRST 3 ROWS ONLY
            """
            params = {"query_vec": json.dumps(query_embedding)}
            for i, ds in enumerate(dataset_filter):
                params[f"ds{i}"] = ds
        else:
            sql = """
                SELECT title, content, dataset
                FROM VECTOR_DOCUMENTS
                ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                FETCH FIRST 3 ROWS ONLY
            """
            params = {"query_vec": json.dumps(query_embedding)}

        cursor.execute(sql, params)

        results = []
        for row in cursor:
            # Read CLOB content
            content = row[1].read() if hasattr(row[1], "read") else str(row[1])
            results.append(f"[{row[2]}] {row[0]}\n{content[:500]}...")

        cursor.close()

        if not results:
            return "No results found."
        return "\n\n---\n\n".join(results)

    search_tool = search

    # Create agent (using ReAct agent for Python 3.14 compatibility)
    print("  Creating research agent...")
    agent = create_oci_agent(
        tools=[search_tool],
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        system_prompt=(
            "You are a research assistant with access to a semantic search tool. "
            "Use it to find relevant information and provide concise answers. "
            "Always cite your sources."
        ),
        temperature=0.3,
        max_tokens=1024,
    )
    print("  Agent created!")

    # Test query
    test_query = "What are common symptoms of diabetes according to medical literature?"
    print(f"\n  Test Query: '{test_query}'")
    print("\n  Agent Response:")
    print("  " + "-" * 50)

    try:
        result = agent.invoke({"messages": [HumanMessage(content=test_query)]})

        # Get response
        final_message = result["messages"][-1]
        response_text = final_message.content

        # Print response (indented)
        for line in response_text.split("\n"):
            print(f"  {line}")

        # Count tool calls
        tool_calls = [
            m for m in result["messages"] if type(m).__name__ == "ToolMessage"
        ]
        print("  " + "-" * 50)
        print(f"  Tool calls made: {len(tool_calls)}")

        conn.close()
        print("\n  [PASS] Deep research agent working correctly")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Agent error: {e}")
        import traceback

        traceback.print_exc()
        conn.close()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DEEP RESEARCH AGENT - COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Database: {DB_DSN[:50]}...")
    print(f"GenAI Region: {GENAI_REGION}")

    results = []

    # Test 1: Database
    results.append(("Database Connectivity", test_database_connection()))

    # Test 2: Vector Search
    results.append(("Vector Search", test_vector_search()))

    # Test 3: Deep Research Agent
    results.append(("Deep Research Agent", test_deep_research_agent()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")
        if result:
            passed += 1

    print(f"\n  Total: {passed}/{len(results)} tests passed")
    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
