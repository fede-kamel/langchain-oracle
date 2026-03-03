#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Download and upload research datasets to OCI Object Storage.

This script downloads medical and legal datasets from HuggingFace
and uploads them to OCI Object Storage buckets for deep research.

Usage:
    python scripts/upload_research_datasets.py

Requirements:
    pip install datasets oci
"""

import json

import oci
from datasets import load_dataset

# OCI Configuration
COMPARTMENT_ID = (
    "ocid1.compartment.oc1.."
    "aaaaaaaaih6r7yauu3rloxnf4gj7uidfr5sbtsanxgnej5zfxu27gsbn6o5a"
)
NAMESPACE = "id0qhv5yj7ke"
REGION = "us-ashburn-1"
PROFILE = "API_KEY_AUTH"

# Bucket names
MEDICAL_BUCKET = "deep-research-medical"
LEGAL_BUCKET = "deep-research-legal"


def get_oci_client():
    """Create OCI Object Storage client."""
    config = oci.config.from_file(profile_name=PROFILE)
    # Override region to ensure we're using the correct one
    config["region"] = REGION
    return oci.object_storage.ObjectStorageClient(config)


def upload_json_to_bucket(client, bucket_name: str, object_name: str, data: list):
    """Upload JSON data to OCI bucket."""
    json_data = json.dumps(data, indent=2, ensure_ascii=False)

    client.put_object(
        namespace_name=NAMESPACE,
        bucket_name=bucket_name,
        object_name=object_name,
        put_object_body=json_data.encode("utf-8"),
        content_type="application/json",
    )
    print(f"  Uploaded {object_name} ({len(data)} records)")


def download_and_upload_medmcqa(client, n_samples: int = 1000):
    """Download MedMCQA dataset and upload to OCI."""
    print("\n=== Downloading MedMCQA (Medical QA) ===")

    dataset = load_dataset(
        "openlifescienceai/medmcqa",
        split="validation",
    )

    # Group by subject
    subjects = {}
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        subject = item.get("subject_name", "Unknown")
        if subject not in subjects:
            subjects[subject] = []

        subjects[subject].append(
            {
                "id": str(item["id"]),
                "question": item["question"],
                "options": {
                    "A": item["opa"],
                    "B": item["opb"],
                    "C": item["opc"],
                    "D": item["opd"],
                },
                "correct_answer": ["A", "B", "C", "D"][item["cop"]],
                "subject": subject,
                "topic": item.get("topic_name", ""),
                "explanation": item.get("exp", ""),
            }
        )

    # Upload each subject as separate file
    print(f"  Found {len(subjects)} subjects")
    for subject, questions in subjects.items():
        safe_name = subject.lower().replace(" ", "_").replace("&", "and")
        object_name = f"medmcqa/{safe_name}.json"
        upload_json_to_bucket(client, MEDICAL_BUCKET, object_name, questions)

    # Upload combined index
    index = {
        "dataset": "MedMCQA",
        "source": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
        "total_samples": sum(len(q) for q in subjects.values()),
        "subjects": {s: len(q) for s, q in subjects.items()},
    }
    upload_json_to_bucket(client, MEDICAL_BUCKET, "medmcqa/index.json", [index])


def download_and_upload_pubmedqa(client, n_samples: int = 500):
    """Download PubMedQA dataset and upload to OCI."""
    print("\n=== Downloading PubMedQA (Biomedical Research) ===")

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        name="pqa_labeled",
        split="train",
    )

    records = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        contexts = item.get("context", {}).get("contexts", [])

        records.append(
            {
                "id": str(item["pubid"]),
                "question": item["question"],
                "context": "\n\n".join(contexts),
                "answer": item.get("final_decision", ""),
                "long_answer": item.get("long_answer", ""),
            }
        )

    upload_json_to_bucket(client, MEDICAL_BUCKET, "pubmedqa/questions.json", records)

    # Upload index
    index = {
        "dataset": "PubMedQA",
        "source": "https://huggingface.co/datasets/qiaojin/PubMedQA",
        "total_samples": len(records),
        "config": "pqa_labeled",
    }
    upload_json_to_bucket(client, MEDICAL_BUCKET, "pubmedqa/index.json", [index])


def download_and_upload_cuad(client, n_samples: int = 500):
    """Download CUAD legal contract dataset and upload to OCI."""
    print("\n=== Downloading CUAD (Legal Contracts) ===")

    try:
        dataset = load_dataset(
            "theatticusproject/cuad-qa",
            split="train",
        )
    except Exception:
        # Fallback
        dataset = load_dataset(
            "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
            split="train",
        )

    records = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        records.append(
            {
                "id": f"cuad_{i}",
                "title": item.get("title", ""),
                "context": item.get("context", item.get("text", ""))[:5000],
                "question": item.get("question", ""),
                "answers": item.get("answers", {}).get("text", []),
            }
        )

    upload_json_to_bucket(client, LEGAL_BUCKET, "cuad/contracts.json", records)

    # Upload index
    index = {
        "dataset": "CUAD",
        "source": "https://huggingface.co/datasets/theatticusproject/cuad-qa",
        "total_samples": len(records),
        "description": "Contract Understanding Atticus Dataset",
    }
    upload_json_to_bucket(client, LEGAL_BUCKET, "cuad/index.json", [index])


def main():
    print("=" * 60)
    print("Uploading Research Datasets to OCI Object Storage")
    print("=" * 60)
    print(f"Compartment: {COMPARTMENT_ID[:50]}...")
    print(f"Namespace: {NAMESPACE}")
    print(f"Region: {REGION}")

    client = get_oci_client()

    # Download and upload each dataset
    download_and_upload_medmcqa(client, n_samples=1000)
    download_and_upload_pubmedqa(client, n_samples=500)
    download_and_upload_cuad(client, n_samples=500)

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)

    # List uploaded objects
    print("\nMedical bucket contents:")
    objects = client.list_objects(NAMESPACE, MEDICAL_BUCKET).data.objects
    for obj in objects:
        print(f"  - {obj.name}")

    print("\nLegal bucket contents:")
    objects = client.list_objects(NAMESPACE, LEGAL_BUCKET).data.objects
    for obj in objects:
        print(f"  - {obj.name}")


if __name__ == "__main__":
    main()
