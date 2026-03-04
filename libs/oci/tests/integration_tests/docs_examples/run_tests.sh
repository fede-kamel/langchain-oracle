#!/bin/bash
# Test runner for OCI documentation examples
#
# This script runs all integration tests for the OCI documentation.
# It uses the API_KEY_AUTH profile from ~/.oci/config

set -e

# Activate langchain-oracle venv
source ~/Projects/langchain-oracle/libs/oci/.venv/bin/activate

# Set environment variables
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q"
export OCI_CONFIG_PROFILE="API_KEY_AUTH"
export OCI_AUTH_TYPE="API_KEY"
export OCI_REGION="us-chicago-1"

# Set model IDs
export OCI_CHAT_MODEL="meta.llama-3.3-70b-instruct"
export OCI_TOOL_MODEL="meta.llama-4-scout-17b-16e-instruct"
export OCI_VISION_MODEL="meta.llama-3.2-90b-vision-instruct"
export OCI_MULTIMODAL_MODEL="google.gemini-2.5-flash"
export OCI_EMBEDDING_MODEL="cohere.embed-english-v3.0"

# Change to test directory
cd "$(dirname "$0")"

# Run tests
echo "Running OCI documentation integration tests..."
echo "Profile: $OCI_CONFIG_PROFILE"
echo "Region: $OCI_REGION"
echo ""

pytest integration_tests/oci/ -v "$@"
