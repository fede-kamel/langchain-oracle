"""Integration tests for OCIGenAIEmbeddings documentation examples.

Tests all code examples from src/oss/python/integrations/text_embedding/oci_generative_ai.mdx

Setup:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_CONFIG_PROFILE="API_KEY_AUTH"
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_REGION="us-chicago-1"

Run:
    pytest tests/integration_tests/oci/test_embeddings_docs_examples.py -v
"""

import numpy as np
import pytest

from langchain_oci import OCIGenAIEmbeddings


@pytest.mark.requires_oci
class TestTextEmbeddings:
    """Test text embedding examples from documentation."""

    def test_code_documentation_search(self, oci_config, embedding_model_id):
        """Test semantic search over technical documentation example."""
        embeddings = OCIGenAIEmbeddings(
            model_id=embedding_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # Index code documentation
        docs = [
            "authenticate() validates JWT tokens and returns user object",
            "authorize() checks user permissions for resource access",
            "audit_log() records user actions for compliance tracking",
        ]
        doc_vectors = embeddings.embed_documents(docs)

        # Verify embeddings
        assert len(doc_vectors) == 3
        for vec in doc_vectors:
            assert len(vec) > 0
            assert isinstance(vec, list)

        # Search with natural language query
        query = "How do I verify user identity?"
        query_vector = embeddings.embed_query(query)

        # Find most relevant documentation
        similarities = [
            np.dot(query_vector, doc_vec)
            / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vec))
            for doc_vec in doc_vectors
        ]
        best_match = docs[np.argmax(similarities)]

        # Should match the authenticate function
        assert "authenticate" in best_match.lower()

    @pytest.mark.asyncio
    async def test_async_embeddings(self, oci_config, embedding_model_id):
        """Test async embedding operations."""
        embeddings = OCIGenAIEmbeddings(
            model_id=embedding_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        query_vector = await embeddings.aembed_query("What is AI?")
        assert len(query_vector) > 0

        doc_vectors = await embeddings.aembed_documents(["Doc 1", "Doc 2"])
        assert len(doc_vectors) == 2


@pytest.mark.requires_oci
@pytest.mark.requires_multimodal
class TestImageEmbeddings:
    """Test image embedding examples from documentation."""

    def test_technical_diagram_search(self, oci_config, tmp_path):
        """Test image embeddings with multimodal model."""
        # Use cohere.embed-v4.0 which supports images
        embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-v4.0",
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # Create a simple test image (1x1 red pixel PNG)
        import base64

        # 1x1 red pixel PNG
        test_image_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx"
            "0gAAAABJRU5ErkJggg=="
        )

        # Write to temp file
        test_image_path = tmp_path / "test.png"
        test_image_path.write_bytes(base64.b64decode(test_image_b64))

        # Test single image embedding
        vector = embeddings.embed_image(str(test_image_path))
        assert len(vector) > 0

        # Test batch embedding
        vectors = embeddings.embed_image_batch([str(test_image_path), str(test_image_path)])
        assert len(vectors) == 2


@pytest.mark.requires_oci
class TestRAGIntegration:
    """Test RAG integration examples from documentation."""

    def test_faiss_integration(self, oci_config, embedding_model_id):
        """Test FAISS vector store integration."""
        pytest.importorskip("faiss")
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        embeddings = OCIGenAIEmbeddings(
            model_id=embedding_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        # Create simple test documents
        documents = [
            Document(page_content="API authentication guide"),
            Document(page_content="Database connection setup"),
            Document(page_content="Error handling best practices"),
        ]

        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Search
        results = vectorstore.similarity_search("authentication", k=1)
        assert len(results) > 0
        assert "authentication" in results[0].page_content.lower()
