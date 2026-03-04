"""Configuration for OCI documentation integration tests.

These tests validate all code examples from the OCI documentation.

Setup:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_CONFIG_PROFILE="API_KEY_AUTH"  # or API_FREE_TIER
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_REGION="us-chicago-1"
"""

import os
from typing import Any, Dict

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_oci: mark test as requiring OCI credentials"
    )
    config.addinivalue_line(
        "markers", "requires_multimodal: mark test as requiring multimodal model"
    )


@pytest.fixture(scope="session")
def oci_config() -> Dict[str, Any]:
    """Get OCI configuration from environment."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = os.environ.get(
        "OCI_GENAI_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )

    return {
        "compartment_id": compartment_id,
        "service_endpoint": endpoint,
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        "region": region,
    }


@pytest.fixture(scope="session")
def chat_model_id() -> str:
    """Get chat model ID from environment."""
    return os.getenv("OCI_CHAT_MODEL", "meta.llama-3.3-70b-instruct")


@pytest.fixture(scope="session")
def vision_model_id() -> str:
    """Get vision model ID from environment."""
    return os.getenv("OCI_VISION_MODEL", "meta.llama-3.2-90b-vision-instruct")


@pytest.fixture(scope="session")
def multimodal_model_id() -> str:
    """Get multimodal model ID (PDF/video/audio) from environment."""
    return os.getenv("OCI_MULTIMODAL_MODEL", "google.gemini-2.5-flash")


@pytest.fixture(scope="session")
def embedding_model_id() -> str:
    """Get embedding model ID from environment."""
    return os.getenv("OCI_EMBEDDING_MODEL", "cohere.embed-english-v3.0")


@pytest.fixture(scope="session")
def tool_calling_model_id() -> str:
    """Get tool-calling model ID from environment."""
    return os.getenv("OCI_TOOL_MODEL", "meta.llama-4-scout-17b-16e-instruct")
