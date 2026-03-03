#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Download and upload large-scale datasets to OCI Object Storage.

This script downloads massive datasets (millions of records) from HuggingFace
and uploads them to OCI Object Storage for deep research.

Datasets:
- Wikipedia: 6.7M+ English articles
- PubMed: 36M+ biomedical citations
- ArXiv: 2.4M+ scientific papers

Usage:
    # Upload Wikipedia (default: 1M articles)
    python scripts/upload_large_datasets.py --dataset wikipedia --samples 1000000

    # Upload PubMed (default: 5M records)
    python scripts/upload_large_datasets.py --dataset pubmed --samples 5000000

    # Upload ArXiv (default: 500K papers)
    python scripts/upload_large_datasets.py --dataset arxiv --samples 500000

    # Upload all datasets
    python scripts/upload_large_datasets.py --dataset all

Requirements:
    pip install datasets oci tqdm
"""

import argparse
import json
from datetime import datetime

import oci
from datasets import load_dataset
from tqdm import tqdm

# OCI Configuration
COMPARTMENT_ID = (
    "ocid1.compartment.oc1.."
    "aaaaaaaaih6r7yauu3rloxnf4gj7uidfr5sbtsanxgnej5zfxu27gsbn6o5a"
)
NAMESPACE = "id0qhv5yj7ke"
REGION = "us-ashburn-1"
PROFILE = "API_KEY_AUTH"

# Bucket for large datasets
LARGE_DATA_BUCKET = "deep-research-large"

# Batch size for uploads (records per file)
BATCH_SIZE = 10000


def get_oci_client():
    """Create OCI Object Storage client."""
    config = oci.config.from_file(profile_name=PROFILE)
    config["region"] = REGION
    return oci.object_storage.ObjectStorageClient(config)


def ensure_bucket_exists(client, bucket_name: str):
    """Create bucket if it doesn't exist."""
    try:
        client.get_bucket(NAMESPACE, bucket_name)
        print(f"  Bucket '{bucket_name}' exists")
    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            print(f"  Creating bucket '{bucket_name}'...")
            client.create_bucket(
                NAMESPACE,
                oci.object_storage.models.CreateBucketDetails(
                    name=bucket_name,
                    compartment_id=COMPARTMENT_ID,
                    storage_tier="Standard",
                ),
            )
            print(f"  Bucket '{bucket_name}' created")
        else:
            raise


def upload_batch(client, bucket: str, prefix: str, batch_num: int, records: list):
    """Upload a batch of records as JSON."""
    object_name = f"{prefix}/batch_{batch_num:06d}.json"
    json_data = json.dumps(records, ensure_ascii=False)

    client.put_object(
        namespace_name=NAMESPACE,
        bucket_name=bucket,
        object_name=object_name,
        put_object_body=json_data.encode("utf-8"),
        content_type="application/json",
    )
    return len(json_data)


def upload_wikipedia(client, n_samples: int = 1_000_000):
    """Upload Wikipedia articles."""
    print(f"\n{'=' * 60}")
    print("Downloading Wikipedia (English)")
    print(f"Target: {n_samples:,} articles")
    print(f"{'=' * 60}")

    # Use wikimedia/wikipedia which is the new format
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,  # Stream to handle large data
    )

    batch = []
    batch_num = 0
    total_uploaded = 0
    total_bytes = 0

    progress = tqdm(total=n_samples, desc="Uploading Wikipedia")

    for i, article in enumerate(dataset):
        if i >= n_samples:
            break

        # Extract relevant fields
        record = {
            "id": article["id"],
            "title": article["title"],
            "text": article["text"][:10000],  # Truncate to 10K chars
            "url": article["url"],
        }
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            bytes_uploaded = upload_batch(
                client, LARGE_DATA_BUCKET, "wikipedia", batch_num, batch
            )
            total_bytes += bytes_uploaded
            total_uploaded += len(batch)
            batch_num += 1
            batch = []
            progress.update(BATCH_SIZE)

    # Upload remaining
    if batch:
        bytes_uploaded = upload_batch(
            client, LARGE_DATA_BUCKET, "wikipedia", batch_num, batch
        )
        total_bytes += bytes_uploaded
        total_uploaded += len(batch)
        progress.update(len(batch))

    progress.close()

    # Upload index
    index = {
        "dataset": "Wikipedia",
        "language": "English",
        "source": "20220301.en",
        "total_records": total_uploaded,
        "total_batches": batch_num + 1,
        "batch_size": BATCH_SIZE,
        "total_bytes": total_bytes,
        "uploaded_at": datetime.now().isoformat(),
    }
    client.put_object(
        namespace_name=NAMESPACE,
        bucket_name=LARGE_DATA_BUCKET,
        object_name="wikipedia/index.json",
        put_object_body=json.dumps(index, indent=2).encode("utf-8"),
        content_type="application/json",
    )

    print(f"\nUploaded {total_uploaded:,} Wikipedia articles")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    return total_uploaded


def upload_pubmed(client, n_samples: int = 5_000_000):
    """Upload C4 dataset (Colossal Clean Crawled Corpus)."""
    print(f"\n{'=' * 60}")
    print("Downloading C4 (Colossal Clean Crawled Corpus)")
    print(f"Target: {n_samples:,} records")
    print(f"{'=' * 60}")

    # C4 dataset: ~364M examples of cleaned web text
    # This is one of the largest high-quality text datasets
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    batch = []
    batch_num = 0
    total_uploaded = 0
    total_bytes = 0

    progress = tqdm(total=n_samples, desc="Uploading C4 Web Text")

    for i, record in enumerate(dataset):
        if i >= n_samples:
            break

        c4_record = {
            "id": f"c4_{i}",
            "text": record.get("text", "")[:10000],  # First 10K chars
            "url": record.get("url", ""),
            "timestamp": record.get("timestamp", ""),
        }

        batch.append(c4_record)

        if len(batch) >= BATCH_SIZE:
            bytes_uploaded = upload_batch(
                client, LARGE_DATA_BUCKET, "c4", batch_num, batch
            )
            total_bytes += bytes_uploaded
            total_uploaded += len(batch)
            batch_num += 1
            batch = []
            progress.update(BATCH_SIZE)

    # Upload remaining
    if batch:
        bytes_uploaded = upload_batch(client, LARGE_DATA_BUCKET, "c4", batch_num, batch)
        total_bytes += bytes_uploaded
        total_uploaded += len(batch)
        progress.update(len(batch))

    progress.close()

    # Upload index
    index = {
        "dataset": "C4 (Colossal Clean Crawled Corpus)",
        "description": "Cleaned web text from Common Crawl",
        "source": "allenai/c4",
        "total_records": total_uploaded,
        "total_batches": batch_num + 1,
        "batch_size": BATCH_SIZE,
        "total_bytes": total_bytes,
        "uploaded_at": datetime.now().isoformat(),
    }
    client.put_object(
        namespace_name=NAMESPACE,
        bucket_name=LARGE_DATA_BUCKET,
        object_name="c4/index.json",
        put_object_body=json.dumps(index, indent=2).encode("utf-8"),
        content_type="application/json",
    )

    print(f"\nUploaded {total_uploaded:,} C4 records")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    return total_uploaded


def upload_arxiv(client, n_samples: int = 500_000):
    """Upload ArXiv papers."""
    print(f"\n{'=' * 60}")
    print("Downloading ArXiv Papers")
    print(f"Target: {n_samples:,} papers")
    print(f"{'=' * 60}")

    # ArXiv dataset
    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True,
    )

    batch = []
    batch_num = 0
    total_uploaded = 0
    total_bytes = 0

    progress = tqdm(total=n_samples, desc="Uploading ArXiv")

    for i, paper in enumerate(dataset):
        if i >= n_samples:
            break

        record = {
            "id": f"arxiv_{i}",
            "title": paper.get("title", "")[:500],
            "abstract": paper.get("abstract", "")[:2000],
            "article": paper.get("article", "")[:15000],  # First 15K chars
        }
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            bytes_uploaded = upload_batch(
                client, LARGE_DATA_BUCKET, "arxiv", batch_num, batch
            )
            total_bytes += bytes_uploaded
            total_uploaded += len(batch)
            batch_num += 1
            batch = []
            progress.update(BATCH_SIZE)

    # Upload remaining
    if batch:
        bytes_uploaded = upload_batch(
            client, LARGE_DATA_BUCKET, "arxiv", batch_num, batch
        )
        total_bytes += bytes_uploaded
        total_uploaded += len(batch)
        progress.update(len(batch))

    progress.close()

    # Upload index
    index = {
        "dataset": "ArXiv",
        "description": "Scientific papers with summaries",
        "source": "ccdv/arxiv-summarization",
        "total_records": total_uploaded,
        "total_batches": batch_num + 1,
        "batch_size": BATCH_SIZE,
        "total_bytes": total_bytes,
        "uploaded_at": datetime.now().isoformat(),
    }
    client.put_object(
        namespace_name=NAMESPACE,
        bucket_name=LARGE_DATA_BUCKET,
        object_name="arxiv/index.json",
        put_object_body=json.dumps(index, indent=2).encode("utf-8"),
        content_type="application/json",
    )

    print(f"\nUploaded {total_uploaded:,} ArXiv papers")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    return total_uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload large datasets to OCI")
    parser.add_argument(
        "--dataset",
        choices=["wikipedia", "pubmed", "arxiv", "all"],
        default="wikipedia",
        help="Dataset to upload",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to upload",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Large-Scale Dataset Upload to OCI Object Storage")
    print("=" * 60)
    print(f"Bucket: {LARGE_DATA_BUCKET}")
    print(f"Region: {REGION}")

    client = get_oci_client()

    # Ensure bucket exists
    ensure_bucket_exists(client, LARGE_DATA_BUCKET)

    total_records = 0

    if args.dataset in ["wikipedia", "all"]:
        samples = args.samples or 1_000_000
        total_records += upload_wikipedia(client, samples)

    if args.dataset in ["pubmed", "all"]:
        samples = args.samples or 5_000_000
        total_records += upload_pubmed(client, samples)

    if args.dataset in ["arxiv", "all"]:
        samples = args.samples or 500_000
        total_records += upload_arxiv(client, samples)

    print("\n" + "=" * 60)
    print(f"Upload complete! Total records: {total_records:,}")
    print("=" * 60)

    # List contents
    print(f"\nBucket '{LARGE_DATA_BUCKET}' contents:")
    response = client.list_objects(NAMESPACE, LARGE_DATA_BUCKET, prefix="")
    for obj in response.data.objects[:20]:
        size_mb = (obj.size or 0) / (1024 * 1024)
        print(f"  - {obj.name} ({size_mb:.1f} MB)")
    if len(response.data.objects) > 20:
        print(f"  ... and {len(response.data.objects) - 20} more files")


if __name__ == "__main__":
    main()
