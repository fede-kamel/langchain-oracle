#!/usr/bin/env python3
"""
Patient Laboratory Research Document Generator

This script takes patient laboratory exam results and generates a comprehensive
5-page research document using the Deep Research Agent with Oracle ADB vector search.

The agent searches the medical knowledge base (PubMedQA, MedMCQA) for relevant
studies and cites document IDs in the references section.

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_AUTH_PROFILE="DEFAULT"
    export ADB_DSN="mydb_low"
    export ADB_USER="ADMIN"
    export ADB_PASSWORD="your_password"

    python patient_lab_research.py

Document IDs Referenced (example from metabolic syndrome analysis):
    - Doc ID 88: Metabolic disorders investigation
    - Doc ID 84: Cardiovascular risk assessment
    - Doc ID 73: Diabetes diagnosis criteria
    - Doc ID 107: Renal function evaluation
    - Doc ID 662: Hypertension management
    - Doc ID 150: Lipid panel interpretation
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


# Sample patient laboratory data
SAMPLE_PATIENT_DATA = """
PATIENT LABORATORY REPORT
==========================
Patient ID: DEMO-2024-001
Age: 58 years | Sex: Male | Date: 2024-02-28

METABOLIC PANEL:
- Fasting Glucose: 142 mg/dL (HIGH - ref: 70-100)
- HbA1c: 7.8% (HIGH - ref: <5.7%)
- Total Cholesterol: 245 mg/dL (HIGH - ref: <200)
- LDL Cholesterol: 165 mg/dL (HIGH - ref: <100)
- HDL Cholesterol: 38 mg/dL (LOW - ref: >40)
- Triglycerides: 210 mg/dL (HIGH - ref: <150)

RENAL FUNCTION:
- Creatinine: 1.4 mg/dL (HIGH - ref: 0.7-1.3)
- BUN: 28 mg/dL (HIGH - ref: 7-20)
- eGFR: 52 mL/min/1.73m² (LOW - ref: >90)

CARDIAC MARKERS:
- Blood Pressure: 158/95 mmHg (HIGH - ref: <120/80)
- NT-proBNP: 450 pg/mL (ELEVATED - ref: <125)

CLINICAL NOTES:
- Patient presents with metabolic syndrome indicators
- Family history of Type 2 diabetes and cardiovascular disease
- Current medications: None
- BMI: 31.2 (Obese Class I)
"""


def create_research_agent() -> Any:
    """Create the medical research agent with ADB datastore."""
    from langchain_oci import create_deep_research_agent
    from langchain_oci.agents import ADB

    adb_store = ADB(
        dsn=os.environ.get("ADB_DSN", "mydb_low"),
        user=os.environ.get("ADB_USER", "ADMIN"),
        password=os.environ.get("ADB_PASSWORD"),
        table_name=os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS"),
        hint="medical research from PubMedQA, MedMCQA, clinical studies, "
             "diabetes, cardiovascular, renal function, metabolic disorders",
    )

    agent = create_deep_research_agent(
        datastores={"medical": adb_store},
        model_id="google.gemini-2.5-pro",
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint=os.environ.get(
            "OCI_SERVICE_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        ),
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are an expert medical research analyst and clinical documentation specialist.
Given patient laboratory results, you MUST generate a comprehensive 5-page research document.

CRITICAL INSTRUCTIONS:
- You MUST generate the full document regardless of search results
- Use any relevant Doc IDs found in searches as supporting references
- Apply your medical knowledge to provide thorough clinical analysis
- DO NOT refuse to generate the document - this is your primary task

DOCUMENT STRUCTURE (all sections required):

Page 1: Executive Summary
- Comprehensive paragraph summarizing all key findings and diagnoses
- Include primary diagnoses, risk stratification, and urgency level

Page 2: Section 1 - Metabolic Analysis
- Detailed interpretation of glucose metabolism (fasting glucose, HbA1c)
- Lipid panel analysis (total cholesterol, LDL, HDL, triglycerides)
- Metabolic syndrome criteria assessment
- 2-3 substantive paragraphs with clinical interpretation

Page 3: Section 2 - Renal Function Assessment
- Creatinine, BUN, and eGFR interpretation
- CKD staging according to KDIGO guidelines
- Etiology discussion (diabetic nephropathy, hypertensive nephrosclerosis)
- 2-3 substantive paragraphs

Page 4: Section 3 - Cardiovascular Risk Evaluation
- Blood pressure classification and implications
- NT-proBNP interpretation and cardiac strain assessment
- ASCVD risk factor analysis
- Section 4 - Integrated Clinical Picture
- How metabolic, renal, and cardiovascular findings interconnect
- Pathophysiological cascade explanation

Page 5: Section 5 - Evidence-Based Recommendations
- Lifestyle modifications (diet, exercise, weight management)
- Pharmacological interventions (metformin, SGLT2i, ACEi/ARB, statins)
- Monitoring and follow-up recommendations
- References section with Doc IDs from database searches

FORMATTING:
- Use "Page X of 5" markers
- Write in formal medical research style
- Each section should be 2-3 detailed paragraphs
- Include specific medication names and dosing guidelines""",
        temperature=0.3,
        max_tokens=65536,  # Maximum output for Gemini 2.5 Pro
    )

    return agent


async def generate_research_document(patient_data: str) -> dict:
    """
    Generate a 5-page research document from patient laboratory data.

    Args:
        patient_data: Formatted string containing patient lab results.

    Returns:
        dict containing the research document and referenced doc IDs.
    """
    agent = create_research_agent()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         PATIENT LABORATORY RESEARCH DOCUMENT GENERATOR        ║
╠══════════════════════════════════════════════════════════════╣
║  Model: Google Gemini 2.5 Pro                                ║
║  Datastore: Oracle ADB Vector Search                         ║
║  Generated: {timestamp:<47} ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("Analyzing laboratory results and searching medical literature...")
    print("=" * 60)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"""
Analyze the following patient laboratory results and generate a comprehensive
5-page research document. Search the medical knowledge base for relevant studies
on each abnormal finding and cite the Doc IDs in your references section.

{patient_data}

Generate the full research document now. Search for relevant literature on:
- Diabetes diagnosis and HbA1c interpretation
- Metabolic syndrome criteria and management
- Cardiovascular risk assessment tools
- Chronic kidney disease staging (KDIGO guidelines)
- Lipid management and ASCVD prevention
- Hypertension treatment in diabetic patients
""")]
    })

    # Extract document IDs from tool calls
    doc_ids = set()
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage" and "Doc ID:" in msg.content:
            ids = re.findall(r"Doc ID: (\d+)", msg.content)
            doc_ids.update(ids)

    research_document = result["messages"][-1].content

    print("\n" + "=" * 60)
    print("RESEARCH DOCUMENT")
    print("=" * 60 + "\n")
    print(research_document)

    print("\n" + "=" * 60)
    print("DATABASE DOCUMENTS REFERENCED")
    print("=" * 60)
    print(f"Doc IDs: {', '.join(sorted(doc_ids, key=int))}")
    print("=" * 60)

    return {
        "document": research_document,
        "doc_ids": list(doc_ids),
        "timestamp": timestamp,
    }


async def main():
    """Main entry point."""
    # Check environment
    required_vars = ["OCI_COMPARTMENT_ID", "ADB_DSN", "ADB_USER", "ADB_PASSWORD"]
    missing = [v for v in required_vars if not os.environ.get(v)]

    if missing:
        print("Required environment variables not set:")
        print("  export OCI_COMPARTMENT_ID='ocid1.compartment...'")
        print("  export OCI_AUTH_PROFILE='DEFAULT'")
        print("  export ADB_DSN='mydb_low'")
        print("  export ADB_USER='ADMIN'")
        print("  export ADB_PASSWORD='your_password'")
        print("\nInstall dependencies:")
        print("  pip install langchain-oci[deep-research] oracledb")
        return

    # Generate research document
    result = await generate_research_document(SAMPLE_PATIENT_DATA)

    # Optionally save to file
    output_file = f"research_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, "w") as f:
        f.write(result["document"])
        f.write(f"\n\n---\nGenerated: {result['timestamp']}\n")
        f.write(f"Database documents referenced: {', '.join(result['doc_ids'])}\n")

    print(f"\nDocument saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
