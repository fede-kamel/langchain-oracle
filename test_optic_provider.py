"""Test with optic's actual LLM provider to find the hang."""

import sys
import time

# Add optic to path
sys.path.insert(0, "/Users/federico.kamelhar/Projects/observai/optic/src")

# Set environment
import os
os.environ["ENVIRONMENT"] = "prod"

from pydantic import BaseModel, Field
from typing import Any
from langchain_core.tools import StructuredTool


# ============================================================================
# Tool schemas (same as optic)
# ============================================================================
class SpecialistInvocation(BaseModel):
    specialist_id: str = Field(..., description="The agent_id of the specialist")
    focus_area: str = Field(..., description="What aspect to focus on")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class InvokeSpecialistsInput(BaseModel):
    invocations: list[SpecialistInvocation] = Field(
        ..., min_length=1, max_length=4, description="List of specialists to invoke"
    )
    timeout_seconds: int = Field(default=300, ge=30, le=600, description="Timeout")


def invoke_specialists_fn(invocations: list[SpecialistInvocation], timeout_seconds: int = 300) -> str:
    return f"Invoked {len(invocations)} specialists"


def create_tools():
    return [
        StructuredTool.from_function(
            func=invoke_specialists_fn,
            name="invoke_specialists",
            description="Invoke specialist agents for focused investigation",
            args_schema=InvokeSpecialistsInput,
        ),
    ]


def test_oci_provider():
    """Test with OCI provider (langchain-oci)."""
    print("\n" + "="*60)
    print("TEST: OCI Provider (langchain-oci)")
    print("="*60)

    from optic.config import Config
    config = Config()

    # Force OCI provider
    config._data["llm"]["provider"] = "oci"

    from optic.llm.providers.oci_genai.provider import OCIGenAIProvider
    provider = OCIGenAIProvider(config)

    print(f"Creating LLM...")
    start = time.time()
    llm = provider.create_llm(model_override="meta.llama-4-maverick-17b-128e-instruct-fp8")
    print(f"LLM created in {time.time() - start:.2f}s")

    tools = create_tools()
    print(f"Binding {len(tools)} tools...")
    start = time.time()
    llm_with_tools = llm.bind_tools(tools)
    print(f"Tools bound in {time.time() - start:.2f}s")

    prompt = "Invoke the wls-health-agent to investigate high GC percentage"
    print(f"Invoking LLM with prompt: {prompt}")
    start = time.time()

    try:
        result = llm_with_tools.invoke(prompt)
        elapsed = time.time() - start
        print(f"Response in {elapsed:.2f}s")
        print(f"Tool calls: {result.tool_calls}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {e}")
        return False


def test_garthok_provider():
    """Test with Garthok provider (OpenAI-compatible)."""
    print("\n" + "="*60)
    print("TEST: Garthok Provider (OpenAI-compatible)")
    print("="*60)

    from optic.config import Config
    config = Config()

    # Config is already loaded from config.prod.yaml (which has garthok + llama-4-maverick)
    print(f"LLM provider from config: {config.llm_provider}")
    print(f"Model from config: {config.openai.model}")

    from optic.llm.providers.garthok import GarthokProvider
    provider = GarthokProvider(config)

    print(f"Creating LLM...")
    start = time.time()
    llm = provider.create_llm()
    print(f"LLM created in {time.time() - start:.2f}s")

    tools = create_tools()
    print(f"Binding {len(tools)} tools...")
    start = time.time()
    llm_with_tools = llm.bind_tools(tools)
    print(f"Tools bound in {time.time() - start:.2f}s")

    prompt = "Invoke the wls-health-agent to investigate high GC percentage"
    print(f"Invoking LLM with prompt: {prompt}")
    start = time.time()

    try:
        result = llm_with_tools.invoke(prompt)
        elapsed = time.time() - start
        print(f"Response in {elapsed:.2f}s")
        print(f"Tool calls: {result.tool_calls}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing optic's LLM providers with Llama 4 Maverick")

    # Test garthok first (more likely to work based on previous tests)
    test_garthok_provider()

    # Then test OCI
    # test_oci_provider()

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
