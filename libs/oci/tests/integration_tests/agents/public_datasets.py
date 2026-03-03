# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Public datasets for testing deep research agents.

This module provides loaders for publicly available datasets from Hugging Face
that can be used to evaluate deep research agents on real-world tasks.

Datasets included:
- Medical: MedMCQA, PubMedQA
- Legal: CUAD (Contract Understanding)

Requirements:
    pip install datasets

Usage:
    from tests.integration_tests.agents.public_datasets import (
        load_medical_qa_samples,
        load_legal_qa_samples,
        load_pubmed_qa_samples,
    )

    # Load 10 medical questions
    medical_tasks = load_medical_qa_samples(n_samples=10)
    for task in medical_tasks:
        result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})

Sources:
- MedMCQA: https://huggingface.co/datasets/openlifescienceai/medmcqa
- PubMedQA: https://huggingface.co/datasets/qiaojin/PubMedQA
- CUAD: https://huggingface.co/datasets/theatticusproject/cuad-qa
"""

from typing import Any


def _check_datasets_installed() -> None:
    """Check if the datasets library is installed."""
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'datasets' library is required for loading public datasets. "
            "Install it with: pip install datasets"
        ) from e


# =============================================================================
# MEDICAL DATASETS
# =============================================================================


def load_medical_qa_samples(
    n_samples: int = 10,
    subject: str | None = None,
    difficulty: str = "validation",
) -> list[dict[str, Any]]:
    """Load medical QA samples from MedMCQA dataset.

    MedMCQA is a large-scale medical multiple-choice QA dataset with 193k+
    questions covering 21 medical subjects from Indian medical entrance exams.

    Args:
        n_samples: Number of samples to load (default: 10)
        subject: Filter by subject (e.g., "Pathology", "Medicine", "Surgery")
        difficulty: Split to use ("train", "validation", "test")

    Returns:
        List of research tasks with query, options, answer, and explanation

    Example:
        >>> tasks = load_medical_qa_samples(n_samples=5, subject="Pathology")
        >>> for task in tasks:
        ...     print(task["query"])
        ...     print(f"Answer: {task['answer']}")
    """
    _check_datasets_installed()
    from datasets import load_dataset

    dataset = load_dataset(
        "openlifescienceai/medmcqa",
        split=difficulty,
        trust_remote_code=True,
    )

    # Filter by subject if specified
    if subject:
        dataset = dataset.filter(lambda x: x["subject_name"].lower() == subject.lower())

    # Convert to research tasks
    tasks = []
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        # Format as research question
        options_text = (
            f"A) {item['opa']}\nB) {item['opb']}\nC) {item['opc']}\nD) {item['opd']}"
        )

        query = (
            f"Medical Question ({item['subject_name']} - {item['topic_name']}):\n\n"
            f"{item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Please explain your reasoning and provide the correct answer."
        )

        correct_option = option_map.get(item["cop"], "Unknown")
        correct_text = [item["opa"], item["opb"], item["opc"], item["opd"]][item["cop"]]

        tasks.append(
            {
                "id": f"medmcqa_{item['id']}",
                "query": query,
                "domain": "Medical",
                "subject": item["subject_name"],
                "topic": item["topic_name"],
                "answer": f"{correct_option}) {correct_text}",
                "explanation": item.get("exp", ""),
                "options": {
                    "A": item["opa"],
                    "B": item["opb"],
                    "C": item["opc"],
                    "D": item["opd"],
                },
                "correct_option": correct_option,
                "source": "MedMCQA",
            }
        )

    return tasks


def load_pubmed_qa_samples(
    n_samples: int = 10,
    config: str = "pqa_labeled",
) -> list[dict[str, Any]]:
    """Load research questions from PubMedQA dataset.

    PubMedQA contains biomedical research questions derived from PubMed
    abstracts with yes/no/maybe answers.

    Args:
        n_samples: Number of samples to load (default: 10)
        config: Dataset configuration ("pqa_labeled", "pqa_artificial")

    Returns:
        List of research tasks with question, context, and answer

    Example:
        >>> tasks = load_pubmed_qa_samples(n_samples=5)
        >>> for task in tasks:
        ...     print(task["query"])
    """
    _check_datasets_installed()
    from datasets import load_dataset

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        name=config,
        split="train",
        trust_remote_code=True,
    )

    tasks = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        # Extract context from nested structure
        contexts = item.get("context", {}).get("contexts", [])
        context_text = "\n\n".join(contexts) if contexts else ""

        # Build research query
        query = (
            f"Research Question:\n{item['question']}\n\n"
            f"Based on the following research context, provide a detailed "
            f"answer (yes/no/maybe) with explanation:\n\n"
            f"Context:\n{context_text[:2000]}..."  # Truncate for brevity
        )

        tasks.append(
            {
                "id": f"pubmedqa_{item['pubid']}",
                "query": query,
                "domain": "Medical Research",
                "question": item["question"],
                "context": context_text,
                "answer": item.get("final_decision", ""),
                "long_answer": item.get("long_answer", ""),
                "source": "PubMedQA",
            }
        )

    return tasks


# =============================================================================
# LEGAL DATASETS
# =============================================================================


def load_legal_qa_samples(
    n_samples: int = 10,
    clause_type: str | None = None,
) -> list[dict[str, Any]]:
    """Load legal contract QA samples from CUAD dataset.

    CUAD (Contract Understanding Atticus Dataset) contains 13,000+ expert
    annotations from 500+ commercial contracts across 41 clause categories.

    Args:
        n_samples: Number of samples to load (default: 10)
        clause_type: Filter by clause type (e.g., "Termination",
            "Indemnification", "Non-Compete")

    Returns:
        List of contract analysis tasks

    Example:
        >>> tasks = load_legal_qa_samples(n_samples=5)
        >>> for task in tasks:
        ...     print(task["query"])
    """
    _check_datasets_installed()
    from datasets import load_dataset

    try:
        dataset = load_dataset(
            "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback to alternative source
        dataset = load_dataset(
            "theatticusproject/cuad-qa",
            split="train",
            trust_remote_code=True,
        )

    # Filter by clause type if specified
    if clause_type and "clause_type" in dataset.column_names:
        dataset = dataset.filter(
            lambda x: clause_type.lower() in x.get("clause_type", "").lower()
        )

    tasks = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        # Build contract analysis query
        clause_text = item.get("clause_text", item.get("text", ""))[:1500]
        clause_category = item.get("clause_type", item.get("category", "General"))

        query = (
            f"Legal Contract Analysis Task:\n\n"
            f"Clause Type: {clause_category}\n\n"
            f"Contract Text:\n{clause_text}\n\n"
            f"Please analyze this contract clause and identify:\n"
            f"1. Key obligations and rights\n"
            f"2. Potential risks or concerns\n"
            f"3. Important terms and conditions\n"
            f"4. Recommendations for review"
        )

        tasks.append(
            {
                "id": f"cuad_{i}",
                "query": query,
                "domain": "Legal",
                "clause_type": clause_category,
                "contract_text": clause_text,
                "file_name": item.get("file_name", ""),
                "source": "CUAD",
            }
        )

    return tasks


# =============================================================================
# COMBINED LOADER
# =============================================================================


def load_research_benchmark(
    domains: list[str] | None = None,
    samples_per_domain: int = 5,
) -> list[dict[str, Any]]:
    """Load a balanced benchmark from multiple domains.

    Args:
        domains: List of domains to include ("medical", "legal", "pubmed")
                 If None, loads from all domains.
        samples_per_domain: Number of samples per domain

    Returns:
        Combined list of research tasks from all domains

    Example:
        >>> benchmark = load_research_benchmark(
        ...     domains=["medical", "legal"], samples_per_domain=10
        ... )
        >>> print(f"Loaded {len(benchmark)} tasks")
    """
    if domains is None:
        domains = ["medical", "legal", "pubmed"]

    all_tasks = []

    if "medical" in domains:
        try:
            medical = load_medical_qa_samples(n_samples=samples_per_domain)
            all_tasks.extend(medical)
        except Exception:
            pass  # Skip if dataset unavailable

    if "pubmed" in domains:
        try:
            pubmed = load_pubmed_qa_samples(n_samples=samples_per_domain)
            all_tasks.extend(pubmed)
        except Exception:
            pass  # Skip if dataset unavailable

    if "legal" in domains:
        try:
            legal = load_legal_qa_samples(n_samples=samples_per_domain)
            all_tasks.extend(legal)
        except Exception:
            pass  # Skip if dataset unavailable

    return all_tasks


# =============================================================================
# MEDICAL SUBJECTS (for reference)
# =============================================================================

MEDICAL_SUBJECTS = [
    "Anesthesia",
    "Anatomy",
    "Biochemistry",
    "Dental",
    "ENT",
    "Forensic Medicine",
    "Gynaecology & Obstetrics",
    "Medicine",
    "Microbiology",
    "Ophthalmology",
    "Orthopaedics",
    "Pathology",
    "Pediatrics",
    "Pharmacology",
    "Physiology",
    "Psychiatry",
    "Radiology",
    "Skin",
    "Preventive & Social Medicine",
    "Surgery",
    "Unknown",
]

# Legal clause types from CUAD
LEGAL_CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Indemnification",
]
