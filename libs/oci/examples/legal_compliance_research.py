#!/usr/bin/env python3
"""
Corporate Legal Compliance Research Document Generator

This script takes a corporate compliance scenario and generates a comprehensive
legal research document using the Deep Research Agent with Oracle ADB vector search
over Wikipedia legal/business content.

The agent searches Wikipedia articles on common law, administrative law,
business ethics, and constitutional principles to generate analysis.

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export ADB_DSN="deepresearch_low"
    export ADB_USER="ADMIN"
    export ADB_PASSWORD="your_password"

    python legal_compliance_research.py
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


# Corporate compliance scenario
COMPLIANCE_CASE = """
CORPORATE COMPLIANCE CASE REPORT
=================================
Case ID: LEGAL-2024-0392 | Date: 2024-03-01
Company: TechVenture Industries, Inc. (Delaware Corporation)
Industry: Software-as-a-Service (SaaS) / Financial Technology

EXECUTIVE SUMMARY OF ISSUES:
A mid-size fintech company faces multiple regulatory and governance challenges
requiring comprehensive legal analysis across administrative law, common law
precedents, constitutional considerations, and business ethics frameworks.

ISSUE 1: REGULATORY COMPLIANCE DISPUTE
- SEC investigation into disclosure practices for AI-powered trading algorithms
- Company claims proprietary trade secrets exempt from full disclosure
- Administrative agency (SEC) seeking expanded regulatory authority
- Question: Balance between regulatory oversight and proprietary protections

ISSUE 2: BOARD GOVERNANCE CONCERNS
- Minority shareholders allege breach of fiduciary duty by board members
- Directors approved related-party transaction with CEO's family business
- Business judgment rule vs. duty of loyalty conflict
- Common law precedents on director liability sought

ISSUE 3: EMPLOYMENT PRACTICES REVIEW
- Class action lawsuit alleging discriminatory AI hiring algorithms
- Administrative agency (EEOC) investigation pending
- Constitutional equal protection considerations
- Business ethics implications for AI governance

ISSUE 4: DATA PRIVACY CONSTITUTIONAL CHALLENGE
- State privacy law conflicts with federal regulations
- Company operates across 50 states with varying requirements
- Constitutional Commerce Clause and Supremacy Clause analysis needed
- International GDPR compliance adding complexity

ISSUE 5: WHISTLEBLOWER RETALIATION CLAIM
- Former employee alleges termination for reporting compliance violations
- Administrative proceedings before DOL
- Common law wrongful termination precedents applicable
- Business ethics and corporate culture assessment required

STAKEHOLDERS:
- Board of Directors (7 members, 2 independent)
- Executive Leadership Team
- 2,500 employees across 12 states
- Institutional investors (60% ownership)
- Retail customers (500,000 active users)
- Regulatory agencies (SEC, EEOC, FTC, State AGs)

REQUESTED ANALYSIS:
Comprehensive legal research document addressing all five issues with:
- Applicable legal frameworks and precedents
- Administrative law considerations
- Constitutional analysis where relevant
- Business ethics evaluation
- Risk assessment and recommendations
"""


def create_legal_agent() -> Any:
    """Create the legal research agent with ADB datastore."""
    from langchain_oci import create_deep_research_agent
    from langchain_oci.agents import ADB

    adb_store = ADB(
        dsn=os.environ.get("ADB_DSN", "deepresearch_low"),
        user=os.environ.get("ADB_USER", "ADMIN"),
        password=os.environ.get("ADB_PASSWORD"),
        table_name=os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS"),
        hint="legal, common law, administrative law, constitution, business ethics, "
             "corporate governance, regulatory compliance, civil liberties, court",
    )

    agent = create_deep_research_agent(
        datastores={"legal": adb_store},
        model_id="google.gemini-2.5-pro",
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint=os.environ.get(
            "OCI_SERVICE_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        ),
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are an expert corporate attorney. Analyze legal cases and write comprehensive research memoranda.

CRITICAL: After searching the database, you MUST write the complete legal memorandum as your final response.
Do NOT use planning tools or create task lists - just search and then write the document.
Apply IRAC methodology. Cite Doc IDs in references.""",
        temperature=0.3,
        max_tokens=65536,
        middleware=[],
    )

    return agent


async def generate_legal_document(case_data: str) -> dict:
    """Generate comprehensive legal research document."""
    agent = create_legal_agent()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CORPORATE LEGAL COMPLIANCE RESEARCH DOCUMENT GENERATOR               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Model: Google Gemini 2.5 Pro                                                ║
║  Max Tokens: 65,536 (~100 pages)                                             ║
║  Datastore: Oracle ADB Vector Search (Wikipedia Legal Content)              ║
║  Generated: {timestamp:<63} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    print("Analyzing corporate compliance case and searching legal literature...")
    print("This may take several minutes due to extensive content generation...")
    print("=" * 80)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"""
Search the legal knowledge base for relevant articles on common law, administrative law,
constitutional law, business ethics, and corporate governance. Then write a comprehensive
legal research memorandum analyzing this corporate compliance case.

{case_data}

Generate a detailed legal research document covering:

1. EXECUTIVE SUMMARY - Overview of legal issues and key findings
2. REGULATORY COMPLIANCE ANALYSIS (Issue 1)
   - Administrative law framework for SEC authority
   - Trade secret protections under common law
   - Balancing test for disclosure requirements
3. BOARD GOVERNANCE ANALYSIS (Issue 2)
   - Fiduciary duty standards under common law
   - Business judgment rule application
   - Related-party transaction scrutiny
4. EMPLOYMENT PRACTICES ANALYSIS (Issue 3)
   - Administrative agency (EEOC) jurisdiction
   - Constitutional equal protection framework
   - AI algorithmic bias legal standards
5. DATA PRIVACY ANALYSIS (Issue 4)
   - Constitutional Commerce Clause analysis
   - Federal preemption under Supremacy Clause
   - Multi-state compliance framework
6. WHISTLEBLOWER CLAIM ANALYSIS (Issue 5)
   - Administrative proceedings framework
   - Common law wrongful termination
   - Statutory whistleblower protections
7. BUSINESS ETHICS EVALUATION
   - Corporate culture assessment
   - Ethical governance frameworks
   - Stakeholder impact analysis
8. RISK ASSESSMENT & RECOMMENDATIONS
   - Litigation risk matrix
   - Compliance remediation priorities
   - Governance improvement recommendations
9. REFERENCES - All Doc IDs from database searches with descriptions

Generate a COMPREHENSIVE legal memorandum that is approximately 60,000 characters (~10,000 words, ~20 pages).

For EACH of the 9 sections, write 6-10 detailed paragraphs covering:
- Full IRAC analysis (Issue, Rule, Application, Conclusion)
- Relevant statutory frameworks and regulatory guidance
- Common law precedents and case law analysis
- Constitutional considerations where applicable
- Risk assessment with likelihood and impact analysis
- Detailed practical recommendations with implementation steps
- Timeline and resource requirements for remediation

This must be an exhaustive legal research document. Do not summarize - write in full detail.
Use ALL available output tokens. Continue writing until you have covered every aspect thoroughly.

CRITICAL: In References, list ALL Doc IDs found with topic descriptions.
""")]
    })

    # Debug output
    print("\n[DEBUG] Messages received:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', None)
        content_len = len(str(content)) if content else 0
        content_preview = str(content)[:200] if content else "None/Empty"
        print(f"  [{i}] {msg_type} (len={content_len}): {content_preview}...")

    # Extract document IDs
    doc_ids = set()
    for msg in result["messages"]:
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and "Doc ID:" in content:
            ids = re.findall(r"Doc ID: (\d+)", content)
            doc_ids.update(ids)

    # Get final document
    last_msg = result["messages"][-1]
    document = getattr(last_msg, 'content', '') or ''

    if not document or len(document) < 100:
        print("\n[DEBUG] Searching for AIMessage with content...")
        for msg in reversed(result["messages"]):
            if type(msg).__name__ == "AIMessage":
                content = getattr(msg, 'content', '')
                if content and len(content) > 100:
                    document = content
                    break

    # Statistics
    char_count = len(document)
    word_count = len(document.split())
    page_estimate = word_count // 500

    print("\n" + "=" * 80)
    print("DOCUMENT GENERATED")
    print("=" * 80)
    print(f"Characters: {char_count:,}")
    print(f"Words: {word_count:,}")
    print(f"Estimated Pages: {page_estimate}")
    print(f"Doc IDs Referenced: {len(doc_ids)}")
    print("=" * 80 + "\n")

    # Save to file
    output_file = f"legal_compliance_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, "w") as f:
        f.write("# Corporate Legal Compliance Research Document\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write("**Model:** Google Gemini 2.5 Pro (max_tokens=65,536)\n")
        f.write("**Datastore:** Oracle ADB Vector Search (Wikipedia Legal Content)\n")
        f.write(f"**Characters:** {char_count:,}\n")
        f.write(f"**Words:** {word_count:,}\n")
        f.write(f"**Estimated Pages:** {page_estimate}\n")
        f.write(f"**Doc IDs Referenced:** {', '.join(sorted(doc_ids, key=lambda x: int(x) if x.isdigit() else 0)) if doc_ids else 'None'}\n\n")
        f.write("---\n\n")
        f.write(document)

    print(f"Document saved to: {output_file}")

    return {
        "document": document,
        "doc_ids": list(doc_ids),
        "char_count": char_count,
        "word_count": word_count,
        "page_estimate": page_estimate,
        "timestamp": timestamp,
    }


async def main():
    """Main entry point."""
    required_vars = ["OCI_COMPARTMENT_ID", "ADB_DSN", "ADB_USER", "ADB_PASSWORD"]
    missing = [v for v in required_vars if not os.environ.get(v)]

    if missing:
        print("Required environment variables not set:")
        print("  export OCI_COMPARTMENT_ID='ocid1.compartment...'")
        print("  export ADB_DSN='deepresearch_low'")
        print("  export ADB_USER='ADMIN'")
        print("  export ADB_PASSWORD='your_password'")
        print("\nInstall dependencies:")
        print("  pip install langchain-oci[deep-research] oracledb")
        return

    await generate_legal_document(COMPLIANCE_CASE)


if __name__ == "__main__":
    asyncio.run(main())
