"""Integration tests for OCI provider documentation examples.

Tests all code examples from src/oss/python/integrations/providers/oci.mdx

Setup:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_CONFIG_PROFILE="API_KEY_AUTH"
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_REGION="us-chicago-1"

Run:
    pytest tests/integration_tests/oci/test_provider_docs_examples.py -v
"""

import base64

import pytest
from langchain.messages import HumanMessage
from langchain.tools import tool

from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings, create_oci_agent, load_image


@pytest.mark.requires_oci
class TestAuthentication:
    """Test authentication examples from documentation."""

    def test_api_key_auth(self, oci_config, chat_model_id):
        """Test API Key authentication example."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type="API_KEY",
            auth_profile=oci_config["auth_profile"],
        )

        response = llm.invoke("Say hello")
        assert response.content


@pytest.mark.requires_oci
class TestEmbeddings:
    """Test embedding examples from documentation."""

    def test_text_embeddings(self, oci_config):
        """Test text embeddings example."""
        embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        vector = embeddings.embed_query("test query")
        assert len(vector) == 1024  # cohere.embed-english-v3.0 dimension

    @pytest.mark.requires_multimodal
    def test_image_embeddings(self, oci_config, tmp_path):
        """Test image embeddings example."""
        embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-v4.0",
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # 1x1 red pixel PNG (base64)
        import base64

        test_image_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx"
            "0gAAAABJRU5ErkJggg=="
        )

        # Write to temp file
        test_image_path = tmp_path / "test.png"
        test_image_path.write_bytes(base64.b64decode(test_image_b64))

        vector = embeddings.embed_image(str(test_image_path))
        assert len(vector) > 0


@pytest.mark.requires_oci
class TestVision:
    """Test vision examples from documentation."""

    def test_architecture_diagram_analysis(self, oci_config, vision_model_id, tmp_path):
        """Test vision with architecture diagram analysis example."""
        llm = ChatOCIGenAI(
            model_id=vision_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # Create a simple 1x1 red pixel PNG
        import base64

        test_image_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx"
            "0gAAAABJRU5ErkJggg=="
        )

        # Write to temp file
        test_image_path = tmp_path / "architecture_diagram.png"
        test_image_path.write_bytes(base64.b64decode(test_image_b64))

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
        Analyze this system architecture diagram and identify:
        1. All microservices and their responsibilities
        2. Data flow between components
        3. External dependencies and APIs
        4. Potential bottlenecks or single points of failure
        """,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{test_image_b64}"},
                },
            ]
        )

        response = llm.invoke([message])
        assert response.content


@pytest.mark.requires_oci
@pytest.mark.requires_multimodal
class TestGeminiMultimodal:
    """Test Gemini multimodal examples from documentation."""

    def test_pdf_contract_extraction(self, oci_config, multimodal_model_id):
        """Test PDF processing with contract extraction example."""
        llm = ChatOCIGenAI(
            model_id=multimodal_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # Create minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 50 >>
stream
BT /F1 24 Tf 100 700 Td (Vendor Contract) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000262 00000 n
0000000361 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
444
%%EOF"""

        pdf_data = base64.b64encode(pdf_content).decode()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
    Extract key contract terms: parties, effective date, termination clauses,
    payment terms, SLA commitments, and liability caps. Return as JSON.
    """,
                },
                {
                    "type": "document_url",
                    "document_url": {
                        "url": f"data:application/pdf;base64,{pdf_data}"
                    },
                },
            ]
        )

        response = llm.invoke([message])
        assert response.content


@pytest.mark.requires_oci
class TestAIAgents:
    """Test AI agent examples from documentation."""

    def test_infrastructure_monitoring_agent(self, oci_config, tool_calling_model_id):
        """Test create_oci_agent with infrastructure monitoring example."""

        @tool
        def query_infrastructure(resource_type: str, region: str) -> dict:
            """Query OCI infrastructure status and health metrics.

            Args:
                resource_type: Type of resource (compute, database, network)
                region: OCI region to query
            """
            # Mock response
            return {
                "status": "healthy",
                "active_instances": 12,
                "cpu_utilization": "45%",
                "alerts": [],
            }

        agent = create_oci_agent(
            model_id=tool_calling_model_id,
            tools=[query_infrastructure],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config["service_endpoint"],
            system_prompt="You are an infrastructure monitoring assistant.",
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Check compute resource health in us-ashburn-1")]}
        )

        # Verify agent executed
        assert "messages" in result
        assert len(result["messages"]) > 0
