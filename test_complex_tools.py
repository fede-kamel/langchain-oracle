"""Test langchain-oci with Llama 4 Maverick and complex tool schemas.

This script tests whether complex Pydantic tool schemas cause hangs
with langchain-oci + Llama 4 Maverick.
"""

import time
from typing import Any
from pydantic import BaseModel, Field
from langchain_oci import ChatOCIGenAI
from langchain_core.tools import tool


# ============================================================================
# Simple tool (should work)
# ============================================================================
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


# ============================================================================
# Complex tool with nested Pydantic models (may hang)
# ============================================================================
class InvocationRequest(BaseModel):
    """Request to invoke a specialist agent."""
    agent_id: str = Field(description="The ID of the specialist agent to invoke")
    focus_area: str = Field(default="", description="Specific area to focus investigation on")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class InvokeSpecialistsInput(BaseModel):
    """Input for invoking multiple specialists."""
    invocations: list[InvocationRequest] = Field(description="List of specialist invocations")
    timeout_seconds: int = Field(default=300, description="Timeout for all invocations")


@tool(args_schema=InvokeSpecialistsInput)
def invoke_specialists(invocations: list[InvocationRequest], timeout_seconds: int = 300) -> str:
    """Invoke one or more specialist agents to investigate specific aspects."""
    return f"Invoked {len(invocations)} specialists with timeout {timeout_seconds}s"


# ============================================================================
# Test runner
# ============================================================================
def test_llm(tools: list, test_name: str, prompt: str):
    """Test LLM with given tools."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaah7ixt2oanvvualoahejm63r66c3pse5u4nd4gzviax7eeeqhrysq",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        model_kwargs={"temperature": 0.2, "max_tokens": 500},
    )

    llm_with_tools = llm.bind_tools(tools)

    print(f"Prompt: {prompt}")
    print(f"Tools: {[t.name for t in tools]}")
    print("Calling LLM...")

    start = time.time()
    try:
        result = llm_with_tools.invoke(prompt)
        elapsed = time.time() - start
        print(f"Response time: {elapsed:.2f}s")
        print(f"Tool calls: {result.tool_calls}")
        print(f"Content: {result.content[:200] if result.content else '(empty)'}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {e}")
        return False


if __name__ == "__main__":
    print("Testing langchain-oci + Llama 4 Maverick with different tool complexities")
    print(f"Model: meta.llama-4-maverick-17b-128e-instruct-fp8")
    print(f"Provider: OCI GenAI (langchain-oci)")

    # Test 1: Simple tool
    test_llm(
        tools=[get_weather],
        test_name="Simple tool (str param)",
        prompt="What's the weather in Paris?"
    )

    # Test 2: Complex tool with nested Pydantic
    test_llm(
        tools=[invoke_specialists],
        test_name="Complex tool (nested Pydantic, dict[str, Any])",
        prompt="Invoke the wls-health-agent specialist to investigate memory issues"
    )

    # Test 3: Both tools
    test_llm(
        tools=[get_weather, invoke_specialists],
        test_name="Both tools",
        prompt="First check weather in NYC, then invoke vm-health-agent"
    )

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
