#!/usr/bin/env python3
"""Test insight tools with automatic backend routing.

Usage:
    cd libs/oci
    python scripts/test_insight_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("INSIGHT TOOLS - AUTO ROUTING TEST")
    print("=" * 60)

    from langchain_oci import OCIGenAIEmbeddings
    from langchain_oci.tools import create_insight_tools, OpenSearchBackend

    OPENSEARCH_HOST = os.environ.get(
        "OPENSEARCH_HOST",
        "https://ai-dev.observ.us-ashburn-1.ocs.oraclecloud.com:9200"
    )
    OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER", "ai_user")
    OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD", "%36${082h_9}1574")

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        compartment_id=os.environ.get(
            "OCI_COMPARTMENT_ID",
            "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
        ),
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
    )

    # Setup test indices
    from opensearchpy import OpenSearch, RequestsHttpConnection

    client = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )

    indices = ["insight-hr", "insight-sales", "insight-eng"]
    for idx in indices:
        if client.indices.exists(index=idx):
            client.indices.delete(index=idx)
        client.indices.create(
            index=idx,
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

    # Insert test docs
    docs = {
        "insight-hr": [
            {"title": "PTO Policy", "content": "Employees get 20 vacation days per year.", "source": "hr_policy"},
            {"title": "Benefits", "content": "Health insurance covers dental and vision.", "source": "hr_benefits"},
        ],
        "insight-sales": [
            {"title": "Q4 Targets", "content": "Revenue target is $2M for Q4.", "source": "sales_targets"},
            {"title": "Commission", "content": "Sales commission is 5% of revenue.", "source": "sales_comp"},
        ],
        "insight-eng": [
            {"title": "Code Review", "content": "All PRs require 2 approvals before merge.", "source": "eng_process"},
            {"title": "Deployment", "content": "Deploy to staging first, then production.", "source": "eng_deploy"},
        ],
    }

    for idx, doc_list in docs.items():
        emb_list = embeddings.embed_documents([d["content"] for d in doc_list])
        for doc, emb in zip(doc_list, emb_list):
            client.index(index=idx, body={**doc, "embedding": emb}, refresh=True)
    print(f"Inserted test docs into {len(indices)} indices")

    # Create insight tools - NO backend param needed for search!
    tools = create_insight_tools(
        backends={
            "hr": OpenSearchBackend(
                endpoint=OPENSEARCH_HOST,
                index_name="insight-hr",
                username=OPENSEARCH_USER,
                password=OPENSEARCH_PASSWORD,
                use_ssl=True,
                verify_certs=False,
                hint="HR policies, PTO, vacation, benefits, employee handbook",
            ),
            "sales": OpenSearchBackend(
                endpoint=OPENSEARCH_HOST,
                index_name="insight-sales",
                username=OPENSEARCH_USER,
                password=OPENSEARCH_PASSWORD,
                use_ssl=True,
                verify_certs=False,
                hint="sales reports, revenue, targets, commission, customers",
            ),
            "engineering": OpenSearchBackend(
                endpoint=OPENSEARCH_HOST,
                index_name="insight-eng",
                username=OPENSEARCH_USER,
                password=OPENSEARCH_PASSWORD,
                use_ssl=True,
                verify_certs=False,
                hint="engineering, code review, PRs, deployment, technical docs",
            ),
        },
        embedding_model=embeddings,
        enable_write=True,
    )

    print(f"\nCreated {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:60]}...")

    # Test auto-routing
    print("\n" + "=" * 60)
    print("AUTO-ROUTING TESTS")
    print("=" * 60)

    search_tool = next(t for t in tools if t.name == "search")

    tests = [
        ("How many vacation days do I get?", "hr"),
        ("What's our Q4 revenue target?", "sales"),
        ("How do I get my PR approved?", "engineering"),
        ("Tell me about health insurance", "hr"),
        ("What's the sales commission rate?", "sales"),
        ("How do I deploy to production?", "engineering"),
    ]

    results = []
    for query, expected in tests:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")

        # Call search - NO backend param!
        result = search_tool._run(query)

        # Check which backend was used (from result text)
        used = None
        for backend_name in ["hr", "sales", "engineering"]:
            if f"from {backend_name}" in result.lower():
                used = backend_name
                break

        status = "PASS" if used == expected else "FAIL"
        print(f"Routed to: {used} -> {status}")
        results.append(used == expected)

    # Cleanup
    for idx in indices:
        client.indices.delete(index=idx)

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} passed")
    print("=" * 60)

    if passed == len(results):
        print("\nAuto-routing works! No backend param needed.")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
