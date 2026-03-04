# OCI Documentation Integration Tests

This directory contains integration tests that validate all code examples from the OCI documentation pages.

## What Gets Tested

- **Chat Models** (`test_chat_docs_examples.py`)
  - Basic invocation and multi-turn conversations
  - Streaming (sync and async)
  - Tool calling with realistic examples
  - Structured output parsing
  - Configuration options

- **Embeddings** (`test_embeddings_docs_examples.py`)
  - Text embeddings
  - Image embeddings (multimodal models)
  - RAG integration with FAISS
  - Async operations

- **Provider Features** (`test_provider_docs_examples.py`)
  - Authentication methods
  - Vision capabilities
  - Gemini multimodal (PDF processing)
  - AI agents with tools

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OCI Credentials

Set up OCI credentials using one of these methods:

#### Option A: Use Free Tier Account (API_FREE_TIER profile)

```bash
export OCI_COMPARTMENT_ID="ocid1.tenancy.oc1..aaaaaaaaqlhpnytg33ztkwrdpq62p5yxx5gn5ltmkah23m7qebwjzc7x3lcq"
export OCI_CONFIG_PROFILE="API_FREE_TIER"
export OCI_AUTH_TYPE="API_KEY"
export OCI_REGION="us-chicago-1"
```

#### Option B: Use API_KEY_AUTH Profile

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q"
export OCI_CONFIG_PROFILE="API_KEY_AUTH"
export OCI_AUTH_TYPE="API_KEY"
export OCI_REGION="us-chicago-1"
```

### 3. Configure Models (Optional)

Override default models:

```bash
export OCI_CHAT_MODEL="meta.llama-3.3-70b-instruct"
export OCI_TOOL_MODEL="meta.llama-4-scout-17b-16e-instruct"
export OCI_VISION_MODEL="meta.llama-3.2-90b-vision-instruct"
export OCI_MULTIMODAL_MODEL="google.gemini-2.5-flash"
export OCI_EMBEDDING_MODEL="cohere.embed-english-v3.0"
```

## Running Tests

### Run All Tests

```bash
cd tests
pytest -v
```

### Run Specific Test Files

```bash
# Chat examples only
pytest integration_tests/oci/test_chat_docs_examples.py -v

# Embeddings examples only
pytest integration_tests/oci/test_embeddings_docs_examples.py -v

# Provider examples only
pytest integration_tests/oci/test_provider_docs_examples.py -v
```

### Run Specific Test Classes

```bash
# Tool calling tests only
pytest integration_tests/oci/test_chat_docs_examples.py::TestToolCalling -v

# Vision tests only
pytest integration_tests/oci/test_provider_docs_examples.py::TestVision -v
```

### Run Specific Tests

```bash
# Single test
pytest integration_tests/oci/test_chat_docs_examples.py::TestBasicInvocation::test_security_code_review -v
```

### Skip Multimodal Tests

Some tests require multimodal-capable models. To skip them:

```bash
pytest -v -m "not requires_multimodal"
```

## Quick Start with langchain-oracle .env

If you have the langchain-oracle repository set up, source the integration test environment:

```bash
cd ~/Projects/langchain-oracle
source .env.integration

cd ~/Projects/langchain-docs/tests
pytest -v
```

## Test Output

Successful test output will look like:

```
tests/integration_tests/oci/test_chat_docs_examples.py::TestBasicInvocation::test_security_code_review PASSED
tests/integration_tests/oci/test_chat_docs_examples.py::TestBasicInvocation::test_multi_turn_conversation PASSED
tests/integration_tests/oci/test_chat_docs_examples.py::TestStreaming::test_sync_streaming PASSED
...
```

## Troubleshooting

### Missing OCI_COMPARTMENT_ID

```
SKIPPED [1] conftest.py:42: OCI_COMPARTMENT_ID not set
```

**Fix:** Export `OCI_COMPARTMENT_ID` environment variable.

### Authentication Errors

```
ServiceError: NotAuthenticated
```

**Fix:**
- Verify `~/.oci/config` contains the profile specified in `OCI_CONFIG_PROFILE`
- Run `oci session authenticate` if using security tokens
- Check that the profile has access to the compartment

### Model Not Available

```
ServiceError: ModelNotFound
```

**Fix:**
- Verify the model is available in your region
- Check the compartment has access to the model
- Try a different model using environment variables

## CI/CD Integration

To run tests in CI/CD, ensure these environment variables are set:

```yaml
env:
  OCI_COMPARTMENT_ID: ${{ secrets.OCI_COMPARTMENT_ID }}
  OCI_CONFIG_PROFILE: ${{ secrets.OCI_CONFIG_PROFILE }}
  OCI_AUTH_TYPE: API_KEY
  OCI_REGION: us-chicago-1
```

You'll also need to provision `~/.oci/config` with credentials.

## Coverage

These tests validate every code block in:
- `src/oss/python/integrations/chat/oci_generative_ai.mdx`
- `src/oss/python/integrations/text_embedding/oci_generative_ai.mdx`
- `src/oss/python/integrations/providers/oci.mdx`

All examples are tested against real OCI services to ensure accuracy.
