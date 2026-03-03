#!/usr/bin/env python3
"""
SRE Service Topology Investigation Document Generator

This script takes service health metrics and topology data, then generates a
comprehensive 5-page incident investigation document using the Deep Research
Agent with OpenSearch vector search over SRE runbooks and documentation.

The agent searches the knowledge base for relevant runbooks, troubleshooting
guides, and past incident patterns, citing document IDs in the references.

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_AUTH_PROFILE="DEFAULT"
    export OPENSEARCH_ENDPOINT="https://your-opensearch:9200"
    export OPENSEARCH_USER="admin"
    export OPENSEARCH_PASSWORD="your_password"

    python sre_topology_investigation.py
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


# Sample service topology incident data (generic microservices)
SAMPLE_INCIDENT_DATA = """
SERVICE TOPOLOGY INCIDENT REPORT
================================
Incident ID: INC-2024-0892
Severity: P1 (Critical)
Start Time: 2024-02-28 14:23:00 UTC
Detection: Automated alerting

AFFECTED SERVICES:
┌─────────────────────────────────────────────────────────────┐
│  [API Gateway] ──► [Auth Service] ──► [User Service]       │
│       │                  │                   │              │
│       ▼                  ▼                   ▼              │
│  [Rate Limiter]    [Token Cache]      [Profile DB]         │
│       │                  │                   │              │
│       └──────────► [Redis Cluster] ◄─────────┘              │
│                          │                                  │
│                    [DEGRADED]                               │
└─────────────────────────────────────────────────────────────┘

METRICS SNAPSHOT:
- API Gateway:
  * Request latency: 2.3s (p99) - CRITICAL (baseline: 150ms)
  * Error rate: 12.4% - CRITICAL (baseline: 0.1%)
  * Throughput: 45% of normal capacity

- Redis Cluster:
  * Memory usage: 94.2% - CRITICAL (threshold: 85%)
  * Connection count: 12,847 - HIGH (normal: 3,000)
  * Eviction rate: 1,247/sec - CRITICAL (normal: 0)
  * Replication lag: 45ms - WARNING (normal: <5ms)

- Auth Service:
  * CPU utilization: 89% - HIGH (baseline: 35%)
  * Heap memory: 78% - WARNING (threshold: 80%)
  * GC pause time: 450ms - CRITICAL (baseline: 15ms)
  * Retry storms detected: Yes

- Kubernetes Cluster:
  * Pod restarts (last hour): 47 - CRITICAL
  * OOMKilled events: 12 pods
  * Pending pods: 8
  * Node memory pressure: 3 nodes affected

CASCADING FAILURE PATTERN:
1. Redis memory exhaustion triggered evictions
2. Cache miss rate increased to 78%
3. Auth service overwhelmed with token regeneration
4. Connection pool exhaustion propagated upstream
5. API Gateway request queuing caused timeout cascade

RECENT CHANGES (last 24h):
- 13:45 UTC: Deployed user-service v2.4.1 (memory optimization)
- 09:00 UTC: Increased rate limit threshold by 20%
- Previous day: Traffic surge preparation configs
"""


def create_sre_agent() -> Any:
    """Create the SRE investigation agent with OpenSearch datastore."""
    from langchain_oci import create_deep_research_agent
    from langchain_oci.agents import OpenSearch

    opensearch_store = OpenSearch(
        endpoint=os.environ.get("OPENSEARCH_ENDPOINT"),
        index_name=os.environ.get("OPENSEARCH_INDEX", "sre_runbooks"),
        username=os.environ.get("OPENSEARCH_USER"),
        password=os.environ.get("OPENSEARCH_PASSWORD"),
        verify_certs=os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        hint="SRE runbooks, incident response, Kubernetes troubleshooting, "
             "memory management, cache failures, cascading failures, "
             "distributed systems, microservices debugging",
    )

    agent = create_deep_research_agent(
        datastores={"sre": opensearch_store},
        model_id="google.gemini-2.5-pro",
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint=os.environ.get(
            "OCI_SERVICE_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        ),
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are an expert Site Reliability Engineer (SRE) analyst.
Given service topology data and incident metrics, generate a comprehensive 5-page
incident investigation document with the following structure:

STRUCTURE:
- Executive Summary (1 paragraph summarizing the incident and root cause)
- Section 1: Service Topology Analysis (affected services, dependencies, blast radius)
- Section 2: Metrics Deep Dive (detailed analysis of each anomalous metric)
- Section 3: Root Cause Analysis (causal chain, contributing factors, timeline)
- Section 4: Cascading Failure Pattern (how the failure propagated through the system)
- Section 5: Remediation & Prevention (immediate fixes, long-term improvements)
- References (list Doc IDs from the runbook database that informed your analysis)

REQUIREMENTS:
- Search the SRE knowledge base for relevant runbooks and incident patterns
- Write in formal technical style with substantive paragraphs
- Each section should be 2-3 detailed paragraphs
- In the References section, list the Doc IDs that provided evidence
- Use page breaks (Page X of 5) for document structure
- Focus on actionable insights and blameless post-mortem principles""",
        temperature=0.3,
        max_tokens=65536,  # Maximum output for Gemini 2.5 Pro
    )

    return agent


async def generate_investigation_document(incident_data: str) -> dict:
    """
    Generate a 5-page investigation document from incident data.

    Args:
        incident_data: Formatted string containing service topology and metrics.

    Returns:
        dict containing the investigation document and referenced doc IDs.
    """
    agent = create_sre_agent()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         SRE SERVICE TOPOLOGY INVESTIGATION GENERATOR         ║
╠══════════════════════════════════════════════════════════════╣
║  Model: Google Gemini 2.5 Pro                                ║
║  Datastore: OpenSearch (SRE Runbooks)                        ║
║  Generated: {timestamp:<47} ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("Analyzing service topology and searching runbook database...")
    print("=" * 60)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"""
Analyze the following service topology incident and generate a comprehensive
5-page investigation document. Search the SRE knowledge base for relevant
runbooks and cite the Doc IDs in your references section.

{incident_data}

Generate the full investigation document now. Search for relevant runbooks on:
- Memory pressure and OOMKill patterns in Kubernetes
- Redis cache eviction and memory management
- Cascading failure patterns in microservices
- Connection pool exhaustion troubleshooting
- GC pause optimization for JVM services
- Rate limiting and traffic surge handling
- Incident response procedures for P1 outages
""")]
    })

    # Extract document IDs from tool calls
    doc_ids = set()
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage":
            # Look for various doc ID patterns
            ids = re.findall(r"Doc ID: ([^\n,]+)", msg.content)
            doc_ids.update(ids)
            # Also look for runbook IDs
            runbook_ids = re.findall(r"(runbook\.[a-z_.]+)", msg.content)
            doc_ids.update(runbook_ids)

    investigation_document = result["messages"][-1].content

    print("\n" + "=" * 60)
    print("INVESTIGATION DOCUMENT")
    print("=" * 60 + "\n")
    print(investigation_document)

    print("\n" + "=" * 60)
    print("RUNBOOKS REFERENCED")
    print("=" * 60)
    print(f"Doc IDs: {', '.join(sorted(doc_ids))}")
    print("=" * 60)

    return {
        "document": investigation_document,
        "doc_ids": list(doc_ids),
        "timestamp": timestamp,
    }


async def main():
    """Main entry point."""
    # Check environment
    required_vars = ["OCI_COMPARTMENT_ID", "OPENSEARCH_ENDPOINT", "OPENSEARCH_PASSWORD"]
    missing = [v for v in required_vars if not os.environ.get(v)]

    if missing:
        print("Required environment variables not set:")
        print("  export OCI_COMPARTMENT_ID='ocid1.compartment...'")
        print("  export OCI_AUTH_PROFILE='DEFAULT'")
        print("  export OPENSEARCH_ENDPOINT='https://your-opensearch:9200'")
        print("  export OPENSEARCH_USER='admin'")
        print("  export OPENSEARCH_PASSWORD='your_password'")
        print("  export OPENSEARCH_INDEX='sre_runbooks'")
        print("\nInstall dependencies:")
        print("  pip install langchain-oci[deep-research] opensearch-py")
        return

    # Generate investigation document
    result = await generate_investigation_document(SAMPLE_INCIDENT_DATA)

    # Optionally save to file
    output_file = f"investigation_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, "w") as f:
        f.write(result["document"])
        f.write(f"\n\n---\nGenerated: {result['timestamp']}\n")
        f.write(f"Runbooks referenced: {', '.join(result['doc_ids'])}\n")

    print(f"\nDocument saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
