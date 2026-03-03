#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201, E501

"""Deep Legal Research - Comprehensive multi-query analysis.

This demonstrates true deep research capabilities:
1. Multiple parallel research queries
2. Cross-referencing multiple domains
3. Comprehensive synthesis with detailed analysis
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import oci
import oracledb
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)
GENAI_REGION = "us-chicago-1"
DB_DSN = "tcps://adb.ca-toronto-1.oraclecloud.com:1522/g4549e8afff78c6_deepresearch_low.adb.oraclecloud.com"
DB_USER = "ADMIN"
DB_PASSWORD = "Research2026Pass#"


def get_clients():
    """Initialize OCI clients."""
    config = oci.config.from_file(profile_name="API_KEY_AUTH")
    config["region"] = GENAI_REGION

    genai_client = GenerativeAiInferenceClient(
        config,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
    )

    db_conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)

    return genai_client, db_conn


def vector_search(
    conn, genai_client, query: str, top_k: int = 15, dataset_filter: tuple | None = None
) -> list[dict]:
    """Perform semantic vector search."""
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

    cursor = conn.cursor()

    if dataset_filter:
        placeholders = ",".join([f":ds{i}" for i in range(len(dataset_filter))])
        sql = f"""
            SELECT title, content, dataset, source,
                   VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
            FROM VECTOR_DOCUMENTS
            WHERE dataset IN ({placeholders})
            ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
            FETCH FIRST :top_k ROWS ONLY
        """
        params = {"query_vec": json.dumps(query_embedding), "top_k": top_k}
        for i, ds in enumerate(dataset_filter):
            params[f"ds{i}"] = ds
    else:
        sql = """
            SELECT title, content, dataset, source,
                   VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
            FROM VECTOR_DOCUMENTS
            ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
            FETCH FIRST :top_k ROWS ONLY
        """
        params = {"query_vec": json.dumps(query_embedding), "top_k": top_k}

    cursor.execute(sql, params)

    results = []
    for row in cursor:
        content = row[1].read() if hasattr(row[1], "read") else str(row[1])
        results.append(
            {
                "title": row[0],
                "content": content,
                "dataset": row[2],
                "source": row[3],
                "similarity": 1 - row[4],
            }
        )

    cursor.close()
    return results


def research_query(
    conn, genai_client, topic: str, query: str, datasets: tuple | None = None
) -> dict:
    """Execute a single research query."""
    print(f"  🔍 Researching: {topic}...")
    results = vector_search(conn, genai_client, query, top_k=8, dataset_filter=datasets)
    return {
        "topic": topic,
        "query": query,
        "results": results,
        "count": len(results),
    }


def parallel_research(conn, genai_client, queries: list[dict]) -> list[dict]:
    """Execute multiple research queries in parallel."""
    print("\n" + "=" * 70)
    print("PARALLEL RESEARCH PHASE - Executing multiple queries simultaneously")
    print("=" * 70)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for q in queries:
            future = executor.submit(
                research_query,
                conn,
                genai_client,
                q["topic"],
                q["query"],
                q.get("datasets"),
            )
            futures.append(future)

        for future in futures:
            results.append(future.result())

    print(f"\n  ✅ Completed {len(results)} parallel research queries\n")
    return results


def synthesize_with_gemini(
    genai_client, case_description: str, research_results: list[dict]
) -> str:
    """Use Gemini to synthesize comprehensive analysis."""
    print("=" * 70)
    print("SYNTHESIS PHASE - Gemini analyzing all research findings")
    print("=" * 70)

    # Build context from research
    research_context = ""
    for r in research_results:
        research_context += f"\n\n### RESEARCH: {r['topic']}\n"
        research_context += f"Query: {r['query']}\n"
        research_context += f"Found {r['count']} relevant documents:\n\n"
        for i, doc in enumerate(
            r["results"][:8], 1
        ):  # Top 8 per topic for exhaustive research
            research_context += f"**Document {i}** [{doc['dataset']}] (relevance: {doc['similarity']:.2f})\n"
            research_context += f"Title: {doc['title']}\n"
            research_context += f"Source: {doc['source']}\n"
            research_context += f"Full Content:\n{doc['content'][:3000]}\n\n"  # Full content for deep analysis

    prompt = f"""You are an expert legal analyst conducting exhaustive deep research for a major law firm.

## CASE FOR ANALYSIS:
{case_description}

## RESEARCH FINDINGS FROM VECTOR DATABASE:
{research_context}

## YOUR TASK:
Provide an EXHAUSTIVE legal analysis (10,000+ words) structured as a formal legal memorandum:

---

# LEGAL MEMORANDUM

## 1. EXECUTIVE SUMMARY (500+ words)
   - Overview of the dispute and parties
   - Key findings and conclusions
   - Risk assessment summary
   - Primary recommendations

## 2. STATEMENT OF FACTS (400+ words)
   - Chronological timeline of events
   - Key contractual provisions
   - Actions of each party

## 3. ISSUES PRESENTED (300+ words)
   - List each legal question to be addressed
   - Frame issues precisely

## 4. SHORT ANSWERS (400+ words)
   - Brief answer to each issue presented

## 5. DETAILED LEGAL ANALYSIS (3000+ words)

   ### 5.1 Material Breach Analysis
   - Definition under Delaware law
   - Application to Change of Control violation
   - Application to Security Failure/Data Breach
   - Analysis of each breach independently
   - Cumulative effect of multiple breaches

   ### 5.2 Contract Termination Rights
   - Termination for Cause vs. Termination for Convenience
   - Inapplicability of Early Termination Fee
   - Waiver of future obligations
   - Delaware case law on termination provisions

   ### 5.3 Indemnification Analysis
   - Scope of indemnification clause
   - What costs are covered
   - Procedural requirements
   - Delaware law on indemnification enforcement

   ### 5.4 Limitation of Liability Analysis
   - General enforceability of LoL clauses
   - Explicit exceptions in the contract
   - Indemnification carve-out analysis
   - Gross negligence carve-out analysis
   - Application to this case

   ### 5.5 Gross Negligence Analysis
   - Delaware standard for gross negligence
   - Application to CloudScale's conduct
   - Evidence supporting gross negligence finding
   - Impact on liability cap

   ### 5.6 HIPAA and Regulatory Considerations
   - HIPAA breach notification requirements
   - Potential regulatory penalties
   - Impact on damages calculation

## 6. EVIDENCE AND RESEARCH CITATIONS (1000+ words)
   - Cite specific documents from research
   - Quote relevant contract language
   - Reference similar clauses from CUAD dataset
   - Analogous cases and precedents
   - Industry standards for cloud security

## 7. DAMAGES ANALYSIS (800+ words)
   - Categories of recoverable damages
   - Direct costs (breach response, legal fees)
   - Regulatory fines (HIPAA penalties)
   - Consequential damages
   - Total exposure calculation
   - Mitigation of damages

## 8. RISK ASSESSMENT MATRIX
   | Claim | Probability | Best Case | Worst Case | Expected Value |
   - For each major claim
   - Include litigation risks
   - Settlement range analysis

## 9. OPPOSING ARGUMENTS AND REBUTTALS (600+ words)
   - CloudScale's likely defenses
   - "Administrative oversight" argument
   - "Unforeseeable attack" argument
   - Liability cap enforcement argument
   - Rebuttal to each defense

## 10. STRATEGIC RECOMMENDATIONS (800+ words)

   ### 10.1 Immediate Actions (24-72 hours)
   - Document preservation
   - Formal notices
   - Regulatory filings

   ### 10.2 Short-Term Strategy (1-4 weeks)
   - Demand letter content
   - Expert engagement
   - Negotiation approach

   ### 10.3 Medium-Term Strategy (1-3 months)
   - Arbitration preparation
   - Discovery priorities
   - Witness preparation

   ### 10.4 Settlement Considerations
   - Minimum acceptable terms
   - Walk-away points
   - Creative settlement structures

## 11. ARBITRATION STRATEGY (500+ words)
   - AAA Commercial Rules considerations
   - Arbitrator selection strategy
   - Key evidence to present
   - Timeline expectations
   - Cost-benefit analysis

## 12. CONCLUSION AND PRIORITY ACTIONS (300+ words)
   - Final assessment
   - Ranked priority actions
   - Decision points

---

Be exhaustive. This is a high-stakes matter worth $2.7M+. Cite research findings extensively.
Use formal legal memorandum style. Include specific contract section references throughout."""

    # Call Gemini
    print("  🤖 Gemini synthesizing comprehensive analysis...")

    from langchain_oci import ChatOCIGenAI

    llm = ChatOCIGenAI(
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        model_kwargs={
            "temperature": 0.3,
            "max_tokens": 65536,  # MAX: 64K output tokens for exhaustive analysis
        },
    )

    response = llm.invoke(prompt)
    return response.content


def main():
    """Run deep legal research."""
    print("=" * 70)
    print("DEEP LEGAL RESEARCH SYSTEM")
    print("Oracle 26ai Vector DB + Gemini 2.5 Pro")
    print("=" * 70)

    # Initialize
    print("\nInitializing connections...")
    genai_client, conn = get_clients()
    print("  ✅ Connected to Oracle 26ai Vector Database")
    print("  ✅ Connected to OCI GenAI (Gemini)")

    # The Legal Case
    case_description = """
## TechVenture Inc. v. CloudScale Solutions
### Complex SaaS Contract Dispute

**PARTIES:**
- Plaintiff: TechVenture Inc. (Series B startup, 50 employees, healthcare tech)
- Defendant: CloudScale Solutions (cloud infrastructure provider, acquired by MegaCorp)

**CONTRACT DETAILS:**
- 3-year SaaS agreement signed January 2024
- Total contract value: $2.4 million ($800K/year)
- Services: Cloud infrastructure, data hosting, security services

**KEY CONTRACT CLAUSES:**
1. Change of Control Clause (Section 12.4): "Either party shall provide written
   notice within 30 days of any change of control, merger, or acquisition.
   Failure to provide notice constitutes a material breach."

2. Termination for Convenience (Section 14.2): "Either party may terminate with
   90 days written notice and payment of 50% of remaining contract value."

3. Termination for Cause (Section 14.3): "Either party may terminate immediately
   upon material breach, including but not limited to: security failures, data
   breaches, service availability below 99.9% SLA."

4. Indemnification (Section 16.1): "Provider shall indemnify, defend, and hold
   harmless Client from any claims, damages, or losses arising from Provider's
   negligence, including data breaches, unauthorized access, or failure to
   maintain industry-standard security practices."

5. Limitation of Liability (Section 16.4): "Total liability shall not exceed
   the fees paid in the 12 months preceding the claim, except for indemnification
   obligations and gross negligence."

**TIMELINE OF EVENTS:**

January 2024: Contract signed
- Standard security assessment passed
- SOC 2 Type II certification confirmed

August 15, 2024: MegaCorp acquires CloudScale
- No notification provided to TechVenture
- TechVenture discovers acquisition through press release on September 1

September 5, 2024: TechVenture sends formal notice
- Requests explanation for failure to notify
- CloudScale responds: "oversight, will provide transition plan"

October 12, 2024: Data breach occurs
- 15,000 TechVenture customer records exposed
- Root cause: Unpatched Apache Struts vulnerability (CVE known since July)
- CloudScale admits delayed patching due to "post-acquisition IT integration"

October 15, 2024: TechVenture discovers breach
- Immediate notification to affected customers required
- Regulatory reporting to HHS (HIPAA) initiated
- Estimated costs: $500K breach response, $200K legal, potential $2M+ regulatory fines

October 20, 2024: TechVenture sends termination notice
- Cites material breach (Change of Control + Security Failure)
- Demands immediate termination without early termination fee
- Invokes indemnification for all breach-related costs

October 25, 2024: CloudScale responds
- Denies material breach characterization
- Demands 50% early termination fee ($1.2M for remaining 2.25 years)
- Offers $100K "goodwill" settlement for breach costs
- Threatens counter-suit for defamation if TechVenture publicizes

**TECHVENTURE'S CLAIMS:**
1. Immediate contract termination without penalty
2. Indemnification for breach costs ($2.7M estimated)
3. Waiver of all remaining contract obligations
4. Declaratory judgment that CloudScale materially breached

**CLOUDSCALE'S POSITION:**
1. Change of Control notice was "administrative oversight"
2. Data breach was "unforeseeable cyber attack"
3. Early termination fee is enforceable
4. Liability capped at $800K (12 months fees)

**JURISDICTION:** Delaware (contract choice of law)
**ARBITRATION:** Contract requires binding arbitration (AAA Commercial Rules)
"""

    print("\n" + "=" * 70)
    print("CASE LOADED: TechVenture Inc. v. CloudScale Solutions")
    print("=" * 70)

    # Define parallel research queries
    research_queries = [
        {
            "topic": "Contract Termination Clauses",
            "query": "contract termination clause material breach immediate termination SaaS agreement",
            "datasets": ("cuad", "c4"),
        },
        {
            "topic": "Change of Control Provisions",
            "query": "change of control clause merger acquisition notification requirement breach",
            "datasets": ("cuad", "c4"),
        },
        {
            "topic": "Data Breach Indemnification",
            "query": "indemnification data breach negligence security failure liability",
            "datasets": ("cuad", "c4"),
        },
        {
            "topic": "Early Termination Fees",
            "query": "early termination fee penalty liquidated damages enforceability",
            "datasets": ("cuad", "c4"),
        },
        {
            "topic": "Limitation of Liability Exceptions",
            "query": "limitation liability cap exceptions gross negligence indemnification",
            "datasets": ("cuad", "c4"),
        },
        {
            "topic": "HIPAA Healthcare Data Breach",
            "query": "healthcare data breach HIPAA compliance security patient records",
            "datasets": ("pubmedqa", "medmcqa", "c4"),
        },
        {
            "topic": "SaaS Service Level Agreements",
            "query": "SaaS SLA service level agreement cloud provider breach remedies",
            "datasets": ("c4", "wikipedia"),
        },
        {
            "topic": "Delaware Contract Law",
            "query": "Delaware contract law interpretation breach damages commercial",
            "datasets": ("c4", "wikipedia"),
        },
    ]

    # Execute parallel research
    research_results = parallel_research(conn, genai_client, research_queries)

    # Display research summary
    print("=" * 70)
    print("RESEARCH SUMMARY")
    print("=" * 70)
    for r in research_results:
        print(f"\n📚 {r['topic']}")
        print(f"   Query: {r['query'][:60]}...")
        print(f"   Found: {r['count']} relevant documents")
        if r["results"]:
            top = r["results"][0]
            print(
                f"   Top result: [{top['dataset']}] {top['title'][:50]}... (sim: {top['similarity']:.2f})"
            )

    # Synthesize with Gemini
    analysis = synthesize_with_gemini(genai_client, case_description, research_results)

    # Output final analysis
    print("\n" + "=" * 70)
    print("COMPREHENSIVE LEGAL ANALYSIS")
    print("=" * 70)
    print(analysis)

    # Statistics
    total_docs = sum(r["count"] for r in research_results)
    print("\n" + "=" * 70)
    print("RESEARCH STATISTICS")
    print("=" * 70)
    print(f"  Total research queries: {len(research_queries)}")
    print(f"  Total documents analyzed: {total_docs}")
    print("  Datasets searched: CUAD (legal), C4 (web), Wikipedia, PubMedQA, MedMCQA")
    print("  Synthesis model: Gemini 2.5 Pro")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
