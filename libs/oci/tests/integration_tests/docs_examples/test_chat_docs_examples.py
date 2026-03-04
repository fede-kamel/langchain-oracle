"""Integration tests for ChatOCIGenAI documentation examples.

Tests all code examples from src/oss/python/integrations/chat/oci_generative_ai.mdx

Setup:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_CONFIG_PROFILE="API_KEY_AUTH"
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_REGION="us-chicago-1"

Run:
    pytest tests/integration_tests/oci/test_chat_docs_examples.py -v
"""

import asyncio
from typing import List, Literal

import pytest
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

from langchain_oci import ChatOCIGenAI


@pytest.mark.requires_oci
class TestBasicInvocation:
    """Test basic invocation examples from documentation."""

    def test_security_code_review(self, oci_config, chat_model_id):
        """Test the security code review example."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        messages = [
            ("system", "You are a code review assistant."),
            (
                "human",
                """Review this Python function for security issues:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return db.execute(query)
```""",
            ),
        ]
        response = llm.invoke(messages)
        assert response.content
        assert "SQL" in response.content or "injection" in response.content.lower()

    def test_multi_turn_conversation(self, oci_config, chat_model_id):
        """Test multi-turn conversation example."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        messages = [
            HumanMessage(content="Analyze error rate spike at 14:30 UTC"),
            AIMessage(
                content="The spike correlates with deploy-v2.1.3. Checking logs..."
            ),
            HumanMessage(content="What was the root cause?"),
        ]

        response = llm.invoke(messages)
        assert response.content
        assert isinstance(response, AIMessage)


@pytest.mark.requires_oci
class TestStreaming:
    """Test streaming examples from documentation."""

    def test_sync_streaming(self, oci_config, chat_model_id):
        """Test synchronous streaming."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        messages = [("human", "Explain async/await in Python in one sentence.")]
        chunks = []
        for chunk in llm.stream(messages):
            chunks.append(chunk.content)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0


@pytest.mark.requires_oci
class TestAsync:
    """Test async examples from documentation."""

    @pytest.mark.asyncio
    async def test_analyze_codebase(self, oci_config, chat_model_id):
        """Test concurrent analysis example."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        files = [
            "def process(): pass",
            "def validate(): pass",
        ]

        tasks = [
            llm.ainvoke(f"Find security vulnerabilities in:\n{code}") for code in files
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        for result in results:
            assert result.content

    @pytest.mark.asyncio
    async def test_async_streaming(self, oci_config, chat_model_id):
        """Test async streaming example."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        chunks = []
        async for chunk in llm.astream("Generate a one-sentence API doc example"):
            chunks.append(chunk.content)

        assert len(chunks) > 0


@pytest.mark.requires_oci
class TestToolCalling:
    """Test tool calling examples from documentation."""

    def test_analytics_tool(self, oci_config, tool_calling_model_id):
        """Test tool calling with analytics example."""

        @tool
        def query_user_analytics(user_id: str, metric: str) -> dict:
            """Query analytics database for user metrics.

            Args:
                user_id: The user ID to query
                metric: Metric name (revenue, sessions, conversions)
            """
            # Mock response
            return {
                "user_id": user_id,
                "metric": metric,
                "value": 1234,
                "period": "last_30_days",
            }

        llm = ChatOCIGenAI(
            model_id=tool_calling_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
        )

        llm_with_tools = llm.bind_tools([query_user_analytics])
        response = llm_with_tools.invoke("What's user 12345's revenue?")

        # Should either call the tool or mention it
        assert response.content or response.tool_calls


@pytest.mark.requires_oci
class TestStructuredOutput:
    """Test structured output examples from documentation."""

    def test_support_ticket_parsing(self, oci_config, chat_model_id):
        """Test structured output with support ticket example."""

        class SupportTicket(BaseModel):
            """Structured representation of a customer support ticket."""

            ticket_id: str
            severity: Literal["low", "medium", "high", "critical"]
            category: str = Field(description="e.g., billing, technical, account")
            description: str
            affected_services: List[str] = Field(description="List of affected service names")

        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            model_kwargs={"temperature": 0.0},  # More deterministic
        )

        structured_llm = llm.with_structured_output(SupportTicket)

        ticket = structured_llm.invoke(
            """
From: customer@example.com
Subject: URGENT - Cannot access production database

Our production API has been returning 500 errors for the past hour.
The database connection pool appears exhausted. This is affecting
our payment processing and user authentication services.

Ticket ID: TKT-12345
"""
        )

        assert isinstance(ticket, SupportTicket)
        assert ticket.severity in ["low", "medium", "high", "critical"]
        assert ticket.category
        assert ticket.description
        # Check that affected_services is a list (model might return list or extract from description)
        assert isinstance(ticket.affected_services, list)


@pytest.mark.requires_oci
class TestConfiguration:
    """Test configuration examples from documentation."""

    def test_model_kwargs(self, oci_config, chat_model_id):
        """Test model configuration with model_kwargs."""
        llm = ChatOCIGenAI(
            model_id=chat_model_id,
            service_endpoint=oci_config["service_endpoint"],
            compartment_id=oci_config["compartment_id"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
            },
        )

        response = llm.invoke("Say hello in one sentence")
        assert response.content
        assert len(response.content) > 0
