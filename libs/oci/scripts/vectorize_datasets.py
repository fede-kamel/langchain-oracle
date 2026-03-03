#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Vectorize datasets from OCI Object Storage into Oracle 26ai.

This script reads documents from OCI buckets and loads them into an
Oracle 26ai Autonomous Database with vector embeddings for semantic search.

Prerequisites:
1. Oracle wallet downloaded to ~/.oracle-wallet/deepresearch
2. OCI credentials configured
3. oracledb package installed

Usage:
    python scripts/vectorize_datasets.py

Environment variables:
    OCI_COMPARTMENT_ID: Compartment for OCI GenAI embeddings (us-chicago-1)
"""

import json
import os
from typing import Any

import oci
from tqdm import tqdm

# Configuration
STORAGE_REGION = "us-ashburn-1"
GENAI_REGION = "us-chicago-1"
NAMESPACE = "id0qhv5yj7ke"
AUTH_PROFILE = "API_KEY_AUTH"

# Buckets
MEDICAL_BUCKET = "deep-research-medical"
LEGAL_BUCKET = "deep-research-legal"
LARGE_BUCKET = "deep-research-large"

# Database configuration (Free Tier ADB in ca-toronto-1)
DB_DSN = "tcps://adb.ca-toronto-1.oraclecloud.com:1522/g4549e8afff78c6_deepresearch_low.adb.oraclecloud.com"
DB_USER = "ADMIN"
DB_PASSWORD = "Research2026Pass#"

# GenAI compartment for embeddings
GENAI_COMPARTMENT = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)

# Batch size for embeddings
EMBEDDING_BATCH_SIZE = 96  # Cohere limit


def get_oci_clients():
    """Create OCI clients for Object Storage and GenAI."""
    config = oci.config.from_file(profile_name=AUTH_PROFILE)

    # Object Storage client (us-ashburn-1)
    storage_config = dict(config)
    storage_config["region"] = STORAGE_REGION
    storage_client = oci.object_storage.ObjectStorageClient(storage_config)

    # GenAI client (us-chicago-1) for embeddings
    genai_config = dict(config)
    genai_config["region"] = GENAI_REGION
    genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        genai_config,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
    )

    return storage_client, genai_client


def get_db_connection():
    """Create Oracle database connection using thin mode with TLS."""
    import oracledb

    # Connect using thin mode with TLS (mTLS disabled on ADB)
    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DB_DSN,
    )


def setup_table(connection):
    """Create the vector table if it doesn't exist."""
    cursor = connection.cursor()

    # Create table with vector column (1024 dims for Cohere embed-english-v3.0)
    create_sql = """
        CREATE TABLE IF NOT EXISTS VECTOR_DOCUMENTS (
            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            title VARCHAR2(1000),
            content CLOB,
            source VARCHAR2(500),
            dataset VARCHAR2(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding VECTOR(1024, FLOAT32)
        )
    """

    try:
        cursor.execute(create_sql)
        connection.commit()
        print("Created VECTOR_DOCUMENTS table")
    except Exception as e:
        if "ORA-00955" in str(e):
            print("VECTOR_DOCUMENTS table already exists")
        else:
            raise

    # Create vector index
    index_sql = """
        CREATE VECTOR INDEX IF NOT EXISTS VECTOR_DOCUMENTS_vec_idx
        ON VECTOR_DOCUMENTS(embedding)
        ORGANIZATION NEIGHBOR PARTITIONS
        WITH DISTANCE COSINE
    """

    try:
        cursor.execute(index_sql)
        connection.commit()
        print("Created vector index")
    except Exception as e:
        if "ORA-00955" in str(e):
            print("Vector index already exists")
        else:
            print(f"Note: Vector index: {e}")

    cursor.close()


def generate_embeddings(
    genai_client,
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings using OCI GenAI Cohere model."""
    from oci.generative_ai_inference.models import (
        EmbedTextDetails,
        OnDemandServingMode,
    )

    embed_request = EmbedTextDetails(
        inputs=texts,
        serving_mode=OnDemandServingMode(model_id="cohere.embed-english-v3.0"),
        compartment_id=GENAI_COMPARTMENT,
        input_type="SEARCH_DOCUMENT",
        truncate="END",
    )

    response = genai_client.embed_text(embed_request)
    return response.data.embeddings


def read_bucket_objects(
    storage_client,
    bucket_name: str,
    prefix: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read JSON objects from a bucket."""
    objects = []

    # List objects
    response = storage_client.list_objects(
        namespace_name=NAMESPACE,
        bucket_name=bucket_name,
        prefix=prefix,
        limit=1000,
    )

    for obj in response.data.objects:
        if not obj.name.endswith(".json"):
            continue

        # Read object
        obj_response = storage_client.get_object(
            namespace_name=NAMESPACE,
            bucket_name=bucket_name,
            object_name=obj.name,
        )

        data = json.loads(obj_response.data.content.decode("utf-8"))
        if isinstance(data, list):
            objects.extend(data)
        else:
            objects.append(data)

        if limit and len(objects) >= limit:
            break

    return objects[:limit] if limit else objects


def process_medical_dataset(
    storage_client,
    genai_client,
    connection,
    max_records: int = 5000,
):
    """Process medical datasets (MedMCQA, PubMedQA)."""
    print("\n" + "=" * 60)
    print("Processing Medical Datasets")
    print("=" * 60)

    cursor = connection.cursor()

    # MedMCQA
    print("\nLoading MedMCQA...")
    medmcqa = read_bucket_objects(
        storage_client, MEDICAL_BUCKET, "medmcqa/", limit=max_records // 2
    )

    docs = []
    for item in medmcqa:
        subject = item.get("subject_name", "Unknown")
        topic = item.get("topic_name", "Q")
        title = f"MedMCQA: {subject} - {topic}"
        content = f"Question: {item.get('question', '')}\n\nOptions:\n"
        for opt in ["opa", "opb", "opc", "opd"]:
            content += f"- {item.get(opt, '')}\n"
        if item.get("exp"):
            content += f"\nExplanation: {item['exp']}"

        docs.append(
            {
                "title": title[:1000],
                "content": content,
                "source": f"medmcqa/{item.get('id', 'unknown')}",
                "dataset": "medmcqa",
            }
        )

    # PubMedQA
    print("Loading PubMedQA...")
    pubmedqa = read_bucket_objects(
        storage_client, MEDICAL_BUCKET, "pubmedqa/", limit=max_records // 2
    )

    for item in pubmedqa:
        title = f"PubMedQA: {item.get('question', '')[:100]}"
        content = f"Question: {item.get('question', '')}\n\n"
        content += f"Context: {item.get('context', '')}\n\n"
        content += f"Answer: {item.get('long_answer', item.get('final_decision', ''))}"

        docs.append(
            {
                "title": title[:1000],
                "content": content,
                "source": f"pubmedqa/{item.get('pubid', 'unknown')}",
                "dataset": "pubmedqa",
            }
        )

    # Generate embeddings and insert
    insert_documents(cursor, genai_client, docs, "Medical")
    connection.commit()
    cursor.close()


def process_legal_dataset(
    storage_client,
    genai_client,
    connection,
    max_records: int = 2000,
):
    """Process legal dataset (CUAD)."""
    print("\n" + "=" * 60)
    print("Processing Legal Dataset (CUAD)")
    print("=" * 60)

    cursor = connection.cursor()

    cuad = read_bucket_objects(storage_client, LEGAL_BUCKET, "cuad/", limit=max_records)

    docs = []
    for item in cuad:
        title = f"CUAD: {item.get('title', 'Contract Clause')}"
        content = f"Context: {item.get('context', '')}\n\n"
        if item.get("question"):
            content += f"Question: {item['question']}\n"
        if item.get("answers"):
            answers = item["answers"]
            if isinstance(answers, dict) and answers.get("text"):
                content += f"Answers: {', '.join(answers['text'])}"

        docs.append(
            {
                "title": title[:1000],
                "content": content,
                "source": f"cuad/{item.get('id', 'unknown')}",
                "dataset": "cuad",
            }
        )

    insert_documents(cursor, genai_client, docs, "Legal")
    connection.commit()
    cursor.close()


def process_large_dataset(
    storage_client,
    genai_client,
    connection,
    dataset: str,
    max_records: int = 10000,
):
    """Process large datasets (Wikipedia, C4)."""
    print("\n" + "=" * 60)
    print(f"Processing Large Dataset: {dataset}")
    print("=" * 60)

    cursor = connection.cursor()

    # Read from large bucket
    items = read_bucket_objects(
        storage_client, LARGE_BUCKET, f"{dataset}/", limit=max_records
    )

    docs = []
    for item in items:
        if dataset == "wikipedia":
            title = item.get("title", "Wikipedia Article")
            content = item.get("text", "")[:8000]  # Truncate for embedding
            source = f"wikipedia/{item.get('id', 'unknown')}"
        elif dataset == "c4":
            title = f"C4: {item.get('url', 'Web Document')[:100]}"
            content = item.get("text", "")[:8000]
            source = f"c4/{item.get('url', 'unknown')}"
        else:
            continue

        docs.append(
            {
                "title": title[:1000],
                "content": content,
                "source": source,
                "dataset": dataset,
            }
        )

    insert_documents(cursor, genai_client, docs, dataset.title())
    connection.commit()
    cursor.close()


def insert_documents(
    cursor,
    genai_client,
    docs: list[dict],
    dataset_name: str,
):
    """Insert documents with embeddings into the database."""
    if not docs:
        print(f"No documents to insert for {dataset_name}")
        return

    print(f"Inserting {len(docs)} {dataset_name} documents...")

    insert_sql = """
        INSERT INTO VECTOR_DOCUMENTS (title, content, source, dataset, embedding)
        VALUES (:title, :content, :source, :dataset, :embedding)
    """

    # Process in batches for embedding
    batch_range = range(0, len(docs), EMBEDDING_BATCH_SIZE)
    for i in tqdm(batch_range, desc=f"Vectorizing {dataset_name}"):
        batch = docs[i : i + EMBEDDING_BATCH_SIZE]

        # Prepare texts for embedding (use content, truncate if needed)
        texts = [doc["content"][:2000] for doc in batch]

        try:
            embeddings = generate_embeddings(genai_client, texts)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            continue

        # Insert each document
        for doc, embedding in zip(batch, embeddings):
            try:
                cursor.execute(
                    insert_sql,
                    {
                        "title": doc["title"],
                        "content": doc["content"],
                        "source": doc["source"],
                        "dataset": doc["dataset"],
                        "embedding": json.dumps(embedding),
                    },
                )
            except Exception as e:
                print(f"Error inserting document: {e}")
                continue


def get_stats(connection):
    """Print database statistics."""
    cursor = connection.cursor()

    print("\n" + "=" * 60)
    print("Vector Database Statistics")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) FROM VECTOR_DOCUMENTS")
    total = cursor.fetchone()[0]
    print(f"Total documents: {total:,}")

    cursor.execute("""
        SELECT dataset, COUNT(*) as cnt
        FROM VECTOR_DOCUMENTS
        GROUP BY dataset
        ORDER BY cnt DESC
    """)

    print("\nBy dataset:")
    for row in cursor:
        print(f"  - {row[0]}: {row[1]:,}")

    cursor.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Vectorizing OCI Object Storage Datasets")
    print("=" * 60)
    print(f"Storage Region: {STORAGE_REGION}")
    print(f"GenAI Region: {GENAI_REGION}")
    print(f"Database: {DB_DSN}")

    # Get clients
    print("\nInitializing clients...")
    storage_client, genai_client = get_oci_clients()

    # Get database connection
    print("Connecting to Oracle 26ai...")
    connection = get_db_connection()

    # Set up table
    print("\nSetting up vector table...")
    setup_table(connection)

    # Process datasets
    process_medical_dataset(storage_client, genai_client, connection, max_records=1000)
    process_legal_dataset(storage_client, genai_client, connection, max_records=500)
    process_large_dataset(
        storage_client, genai_client, connection, "wikipedia", max_records=5000
    )
    process_large_dataset(
        storage_client, genai_client, connection, "c4", max_records=5000
    )

    # Print stats
    get_stats(connection)

    connection.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
