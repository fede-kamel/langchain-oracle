#!/usr/bin/env python3
"""
Neurology Patient Research Document Generator

This script takes comprehensive neurological examination data and generates
an extensive research document (~100 pages) using the Deep Research Agent
with Oracle ADB vector search over medical literature.

The agent searches PubMedQA, MedMCQA for relevant neurological studies
and generates a detailed differential diagnosis, workup, and treatment plan.

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export ADB_DSN="mydb_low"
    export ADB_USER="ADMIN"
    export ADB_PASSWORD="your_password"

    python neurology_research.py
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


# Comprehensive neurology patient case
NEUROLOGY_PATIENT_DATA = """
NEUROLOGICAL CASE REPORT
========================
Patient ID: NEURO-2024-0847 | Age: 42F | Date: 2024-02-28

CHIEF COMPLAINT: Progressive leg weakness, hand numbness, difficulty walking x 3 months

HISTORY OF PRESENT ILLNESS:
- Month 1: Bilateral fingertip paresthesias, Lhermitte's sign, fatigue
- Month 2: Leg weakness on stairs, blurred vision R eye x3 days (resolved), urinary urgency
- Month 3: Wide-based gait, "MS hug" (band-like chest tightness), L eye visual disturbance,
  Uhthoff phenomenon (heat sensitivity), cognitive "brain fog", depression

PAST MEDICAL HISTORY:
- Optic neuritis R eye age 35 (7 years ago) - IV steroids, full recovery
  MRI at that time: "2-3 nonspecific white matter lesions" - no follow-up
- Migraines with aura since age 25
- Vitamin D deficiency (on 2000 IU/day)
- Anxiety (sertraline 50mg)

FAMILY HISTORY: Maternal aunt with MS (diagnosed age 38), Sister with SLE

NEUROLOGICAL EXAMINATION:
- Mental Status: MoCA 24/30, mild word-finding difficulty
- Cranial Nerves: RAPD R eye, temporal disc pallor R, color vision impaired R (8/14 Ishihara)
- Motor: Spastic tone legs R>L, pyramidal weakness legs>arms (4/5), R>L
- Reflexes: Hyperreflexia, bilateral Babinski, R ankle clonus (4 beats), absent abdominal reflexes
- Sensory: Decreased vibration/proprioception feet, sensory level T6-T8
- Coordination: Intention tremor, heel-to-shin impaired, truncal ataxia
- Gait: Wide-based, spastic-ataxic, Romberg positive, tandem unable

LABORATORY FINDINGS:
- CBC, CMP, TSH, B12, Folate: Normal
- Vitamin D: 28 ng/mL (low)
- ANA: Positive 1:80 speckled
- Anti-AQP4, Anti-MOG: Negative
- HIV, RPR, Lyme: Negative

CSF ANALYSIS:
- WBC: 8 cells/μL (95% lymphocytes) - mildly elevated
- Protein: 52 mg/dL - mildly elevated
- Glucose: 58 mg/dL - normal
- IgG Index: 0.92 - ELEVATED
- Oligoclonal Bands: 8 bands CSF-specific - POSITIVE
- Myelin Basic Protein: 4.2 ng/mL - mildly elevated
- All cultures/PCR: Negative

MRI BRAIN (3T with contrast):
- >15 T2/FLAIR hyperintense white matter lesions
- Periventricular lesions perpendicular to ventricles (Dawson's fingers)
- Juxtacortical lesions (4), corpus callosum lesions (3)
- Infratentorial: R middle cerebellar peduncle (2), L pons (1)
- 3 enhancing lesions (2 ring, 1 nodular) - ACTIVE DISEASE
- R optic nerve hyperintensity and atrophy

MRI SPINE (with contrast):
- Cervical: T2 lesion C3-C4 (8mm), enhancing - active
- Thoracic: T2 lesion T6-T7 (6mm), non-enhancing - old

EVOKED POTENTIALS:
- VEP: Prolonged P100 latency R eye (128ms) - demyelination
- SSEP: Prolonged tibial latencies bilaterally
- BAEP: Normal
- EMG/NCS: Normal (no peripheral neuropathy)

OCT: RNFL thinning R eye (68μm) - prior optic neuritis

2017 McDONALD CRITERIA:
✓ Dissemination in Space: Periventricular + Juxtacortical + Infratentorial + Spinal cord
✓ Dissemination in Time: Enhancing + non-enhancing lesions, prior optic neuritis
✓ CSF: Oligoclonal bands positive

DIAGNOSIS: Relapsing-Remitting Multiple Sclerosis (RRMS) with active disease
"""


def create_neurology_agent() -> Any:
    """Create the neurology research agent with ADB datastore."""
    from langchain_oci import create_deep_research_agent
    from langchain_oci.agents import ADB

    adb_store = ADB(
        dsn=os.environ.get("ADB_DSN", "mydb_low"),
        user=os.environ.get("ADB_USER", "ADMIN"),
        password=os.environ.get("ADB_PASSWORD"),
        table_name=os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS"),
        hint="neurology, multiple sclerosis, demyelinating diseases, optic neuritis, "
             "spinal cord lesions, CSF analysis, oligoclonal bands, MRI findings, "
             "evoked potentials, autoimmune disorders, immunotherapy, DMT",
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
        system_prompt="""You are an expert neuroimmunologist. Analyze neurological cases and write comprehensive research documents.

CRITICAL INSTRUCTIONS:
1. Search the medical database for relevant studies
2. ALWAYS cite the Doc IDs you find in your references section, even if the results are general
3. Write the complete research document as your final response
4. In the References section, list ALL Doc IDs found during searches with brief topic descriptions
5. Do NOT say you couldn't access the database - you CAN and MUST use the search results""",
        temperature=0.4,
        max_tokens=65536,  # Maximum output for Gemini 2.5 Pro (~100 pages)
        middleware=[],  # Disable default middleware to simplify flow
    )

    return agent


async def generate_neurology_document(patient_data: str) -> dict:
    """Generate extensive neurology research document."""
    agent = create_neurology_agent()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE NEUROLOGY RESEARCH DOCUMENT GENERATOR                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Model: Google Gemini 2.5 Pro                                                ║
║  Max Tokens: 65,536 (~100 pages)                                             ║
║  Datastore: Oracle ADB Vector Search (PubMedQA, MedMCQA)                     ║
║  Generated: {timestamp:<63} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    print("Analyzing neurological case and searching medical literature...")
    print("This may take several minutes due to extensive content generation...")
    print("=" * 80)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"""
Search the medical knowledge base for relevant studies on multiple sclerosis, demyelinating diseases,
DMT therapy, and neuroimaging. Then write an EXTENSIVE research document analyzing this case.

{patient_data}

Generate a COMPREHENSIVE document (~50,000 words) with detailed analysis in each section:

1. EXECUTIVE SUMMARY (2-3 pages) - Complete overview of findings
2. CLINICAL PRESENTATION (5+ pages) - Detailed symptom timeline, anatomical correlations
3. NEUROLOGICAL EXAMINATION (5+ pages) - Detailed analysis of each finding
4. LABORATORY ANALYSIS (5+ pages) - CSF interpretation, autoantibodies, biomarkers
5. NEUROIMAGING (10+ pages) - Complete MRI analysis, lesion characterization, McDonald criteria
6. EVOKED POTENTIALS (3+ pages) - VEP, SSEP, BAEP interpretation
7. DIFFERENTIAL DIAGNOSIS (10+ pages) - MS vs NMOSD vs MOG-AD, detailed comparison
8. DEFINITIVE DIAGNOSIS (5+ pages) - 2017 McDonald criteria application, EDSS scoring
9. ACUTE TREATMENT (5+ pages) - Methylprednisolone protocols, PLEX considerations
10. DISEASE-MODIFYING THERAPY (15+ pages) - All DMT options, mechanism, efficacy data
11. SYMPTOMATIC MANAGEMENT (10+ pages) - Spasticity, fatigue, bladder, pain, cognitive
12. PROGNOSIS & MONITORING (5+ pages) - Risk factors, MRI surveillance, JCV testing
13. REFERENCES - All Doc IDs from database searches

Write EXTENSIVELY with multiple detailed paragraphs per section. Include specific dosages,
monitoring protocols, and clinical trial data. Use all available output tokens.

CRITICAL: In the References section, you MUST list ALL Doc IDs found during your database searches.
Format each reference as: [Doc ID: X] - Topic/description from the search result.
""")]
    })

    # Debug: Print all messages in detail
    print("\n[DEBUG] Messages received:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', None)
        content_len = len(str(content)) if content else 0
        content_preview = str(content)[:200] if content else "None/Empty"
        print(f"  [{i}] {msg_type} (len={content_len}): {content_preview}...")

    # Extract document IDs from all messages
    doc_ids = set()
    for msg in result["messages"]:
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and "Doc ID:" in content:
            ids = re.findall(r"Doc ID: (\d+)", content)
            doc_ids.update(ids)

    # Get the final document content
    # The last message should be an AIMessage with the research document
    last_msg = result["messages"][-1]
    document = getattr(last_msg, 'content', '') or ''

    # If empty, try to find any AIMessage with substantial content
    if not document or len(document) < 100:
        print("\n[DEBUG] Last message empty or short, searching for AIMessage with content...")
        for msg in reversed(result["messages"]):
            if type(msg).__name__ == "AIMessage":
                content = getattr(msg, 'content', '')
                if content and len(content) > 100:
                    document = content
                    print(f"[DEBUG] Found AIMessage with {len(content)} chars")
                    break

    if not document:
        print("\n[WARNING] No document content found in response!")

    # Calculate statistics
    char_count = len(document)
    word_count = len(document.split())
    page_estimate = word_count // 500  # ~500 words per page

    print("\n" + "=" * 80)
    print("DOCUMENT GENERATED")
    print("=" * 80)
    print(f"Characters: {char_count:,}")
    print(f"Words: {word_count:,}")
    print(f"Estimated Pages: {page_estimate}")
    print(f"Doc IDs Referenced: {len(doc_ids)}")
    print("=" * 80 + "\n")

    # Save to file
    output_file = f"neurology_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, "w") as f:
        f.write(f"# Comprehensive Neurology Research Document\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Model:** Google Gemini 2.5 Pro (max_tokens=65,536)\n")
        f.write(f"**Characters:** {char_count:,}\n")
        f.write(f"**Words:** {word_count:,}\n")
        f.write(f"**Estimated Pages:** {page_estimate}\n")
        f.write(f"**Doc IDs Referenced:** {', '.join(sorted(doc_ids, key=int)) if doc_ids else 'None'}\n\n")
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
        print("  export ADB_DSN='mydb_low'")
        print("  export ADB_USER='ADMIN'")
        print("  export ADB_PASSWORD='your_password'")
        print("\nInstall dependencies:")
        print("  pip install langchain-oci[deep-research] oracledb")
        return

    await generate_neurology_document(NEUROLOGY_PATIENT_DATA)


if __name__ == "__main__":
    asyncio.run(main())
