"""Test with optic's OCI provider to find the hang."""

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
# All 5 orchestrator tools
# ============================================================================
class SearchSpecialistInput(BaseModel):
    query: str = Field(..., description="Search query for specialists")
    capability_filter: str | None = Field(default=None, description="Filter by capability")
    max_results: int = Field(default=5, ge=1, le=10, description="Max results")


class SpecialistInvocation(BaseModel):
    specialist_id: str = Field(..., description="The agent_id of the specialist")
    focus_area: str = Field(..., description="What aspect to focus on")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class InvokeSpecialistsInput(BaseModel):
    invocations: list[SpecialistInvocation] = Field(
        ..., min_length=1, max_length=4, description="List of specialists to invoke"
    )
    timeout_seconds: int = Field(default=300, ge=30, le=600, description="Timeout")


class EvaluateProgressInput(BaseModel):
    force_complete: bool = Field(default=False, description="Force completion")


class SubmitInvestigationInput(BaseModel):
    diagnosis: str = Field(..., description="Root cause diagnosis")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")
    findings: list[str] = Field(default_factory=list, description="Key findings")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    reasoning: str = Field(default="", description="Chain of reasoning")


class CorrelateAndAnalyzeInput(BaseModel):
    entity_labels: dict[str, str] = Field(..., description="Labels to correlate")
    time_range_minutes: int = Field(default=30, ge=5, le=120, description="Time range")
    correlation_strategy: str = Field(default="temporal", description="Strategy")


def dummy_fn(*args, **kwargs):
    return "OK"


def create_all_tools():
    return [
        StructuredTool.from_function(func=dummy_fn, name="search_specialist",
            description="Search for specialist agents", args_schema=SearchSpecialistInput),
        StructuredTool.from_function(func=dummy_fn, name="invoke_specialists",
            description="Invoke specialists", args_schema=InvokeSpecialistsInput),
        StructuredTool.from_function(func=dummy_fn, name="evaluate_progress",
            description="Evaluate progress", args_schema=EvaluateProgressInput),
        StructuredTool.from_function(func=dummy_fn, name="submit_investigation",
            description="Submit investigation", args_schema=SubmitInvestigationInput),
        StructuredTool.from_function(func=dummy_fn, name="correlate_and_analyze",
            description="Correlate and analyze", args_schema=CorrelateAndAnalyzeInput),
    ]


def test_oci_provider():
    """Test with OCI provider (langchain-oci) WITH schema sanitizer."""
    print("\n" + "="*60)
    print("TEST: OCI Provider WITH Schema Sanitizer")
    print("="*60)

    from optic.config import Config
    config = Config()

    # Check OCI config
    print(f"OCI config available: {config.oci_genai is not None}")
    if config.oci_genai:
        print(f"OCI model: {config.oci_genai.model_id}")
        print(f"OCI endpoint: {config.oci_genai.service_endpoint}")
        print(f"OCI auth_profile: {config.oci_genai.auth_profile}")

    from optic.llm.providers.oci_genai.provider import OCIGenAIProvider
    provider = OCIGenAIProvider(config)

    print(f"Creating LLM with schema sanitizer...")
    start = time.time()
    llm = provider.create_llm(model_override="meta.llama-4-maverick-17b-128e-instruct-fp8")
    print(f"LLM created in {time.time() - start:.2f}s")

    tools = create_all_tools()
    print(f"Binding {len(tools)} tools: {[t.name for t in tools]}")
    start = time.time()
    llm_with_tools = llm.bind_tools(tools)
    print(f"Tools bound in {time.time() - start:.2f}s")

    prompt = """You are investigating an anomaly: high GC percentage (3.68%) on WebLogic server UIServer_as1_01.
Invoke the wls-health-agent specialist to investigate this issue."""

    print(f"Invoking LLM (timeout 60s)...")
    start = time.time()

    try:
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(
                asyncio.to_thread(llm_with_tools.invoke, prompt),
                timeout=60
            )
        )
        elapsed = time.time() - start
        print(f"Response in {elapsed:.2f}s")
        print(f"Tool calls: {len(result.tool_calls)} - {[tc['name'] for tc in result.tool_calls]}")
        return True
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing optic's OCI provider with Llama 4 Maverick")
    test_oci_provider()
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
