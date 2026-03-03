#!/bin/bash
# Integration test runner for Deep Research Agent

# Source environment
export OCI_REGION=us-chicago-1
export OCI_SERVICE_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
export OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..aaaaaaaaqlhpnytg33ztkwrdpq62p5yxx5gn5ltmkah23m7qebwjzc7x3lcq
export OCI_AUTH_TYPE=API_KEY
export OCI_AUTH_PROFILE=API_FREE_TIER
export OCI_CHAT_MODEL=meta.llama-3.3-70b-instruct
export OCI_DEEP_RESEARCH_MODEL=meta.llama-3.3-70b-instruct

# OpenSearch
export OPENSEARCH_ENDPOINT=https://ai-dev.observ.us-ashburn-1.ocs.oraclecloud.com:9200
export OPENSEARCH_INDEX=observai_diagnostic-patterns
export OPENSEARCH_USERNAME=ai_user
export OPENSEARCH_PASSWORD='%36${082h_9}1574'
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
export OPENSEARCH_VECTOR_FIELD=vector_field
export OPENSEARCH_SEARCH_FIELDS=text,metadata.title,metadata.content
export OPENSEARCH_HINT="SRE investigations, diagnostic patterns, incidents"
export OPENSEARCH_EMBEDDING_MODEL=cohere.embed-v4.0

# ADB
export ADB_DSN=deepresearch_low
export ADB_WALLET_LOCATION=$HOME/.langchain-oracle-wallet
export ADB_USER=ADMIN
export ADB_PASSWORD='Research2026Pass#'
export ADB_TABLE_NAME=VECTOR_DOCUMENTS
export ADB_HINT="medical research documents"
export ADB_EMBEDDING_MODEL=cohere.embed-v4.0

# Run tests
poetry run pytest "$@"
