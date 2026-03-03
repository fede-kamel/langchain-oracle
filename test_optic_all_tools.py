"""Test with optic's actual provider and ALL orchestrator tools."""

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
# Tool 1: search_specialist
# ============================================================================
class SearchSpecialistInput(BaseModel):
    query: str = Field(..., description="Search query for specialists")
    capability_filter: str | None = Field(default=None, description="Filter by capability")
    max_results: int = Field(default=5, ge=1, le=10, description="Max results")


def search_specialist_fn(query: str, capability_filter: str | None = None, max_results: int = 5) -> str:
    return f"Found specialists for: {query}"


# ============================================================================
# Tool 2: invoke_specialists
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


# ============================================================================
# Tool 3: evaluate_progress
# ============================================================================
class EvaluateProgressInput(BaseModel):
    force_complete: bool = Field(default=False, description="Force completion")


def evaluate_progress_fn(force_complete: bool = False) -> str:
    return "Progress evaluated"


# ============================================================================
# Tool 4: submit_investigation
# ============================================================================
class SubmitInvestigationInput(BaseModel):
    diagnosis: str = Field(..., description="Root cause diagnosis")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")
    findings: list[str] = Field(default_factory=list, description="Key findings")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    reasoning: str = Field(default="", description="Chain of reasoning")


def submit_investigation_fn(
    diagnosis: str,
    confidence: float = 0.5,
    findings: list[str] = None,
    recommendations: list[str] = None,
    reasoning: str = "",
) -> str:
    return f"Submitted: {diagnosis}"


# ============================================================================
# Tool 5: correlate_and_analyze
# ============================================================================
class CorrelateAndAnalyzeInput(BaseModel):
    entity_labels: dict[str, str] = Field(..., description="Labels to correlate")
    time_range_minutes: int = Field(default=30, ge=5, le=120, description="Time range")
    correlation_strategy: str = Field(default="temporal", description="Strategy")


def correlate_and_analyze_fn(
    entity_labels: dict[str, str],
    time_range_minutes: int = 30,
    correlation_strategy: str = "temporal",
) -> str:
    return f"Correlated {len(entity_labels)} labels"


def create_all_tools():
    return [
        StructuredTool.from_function(
            func=search_specialist_fn,
            name="search_specialist",
            description="Search for specialist agents by capability",
            args_schema=SearchSpecialistInput,
        ),
        StructuredTool.from_function(
            func=invoke_specialists_fn,
            name="invoke_specialists",
            description="Invoke specialist agents for focused investigation",
            args_schema=InvokeSpecialistsInput,
        ),
        StructuredTool.from_function(
            func=evaluate_progress_fn,
            name="evaluate_progress",
            description="Evaluate investigation progress",
            args_schema=EvaluateProgressInput,
        ),
        StructuredTool.from_function(
            func=submit_investigation_fn,
            name="submit_investigation",
            description="Submit final investigation results",
            args_schema=SubmitInvestigationInput,
        ),
        StructuredTool.from_function(
            func=correlate_and_analyze_fn,
            name="correlate_and_analyze",
            description="Correlate metrics and analyze patterns",
            args_schema=CorrelateAndAnalyzeInput,
        ),
    ]


def test_garthok_with_all_tools():
    """Test with Garthok provider and all 5 orchestrator tools."""
    print("\n" + "="*60)
    print("TEST: Garthok + ALL 5 Orchestrator Tools")
    print("="*60)

    from optic.config import Config
    config = Config()

    print(f"LLM provider: {config.llm_provider}")
    print(f"Model: {config.openai.model}")

    from optic.llm.providers.garthok import GarthokProvider
    provider = GarthokProvider(config)

    print(f"Creating LLM...")
    start = time.time()
    llm = provider.create_llm()
    print(f"LLM created in {time.time() - start:.2f}s")

    tools = create_all_tools()
    print(f"Binding {len(tools)} tools: {[t.name for t in tools]}")
    start = time.time()
    llm_with_tools = llm.bind_tools(tools)
    print(f"Tools bound in {time.time() - start:.2f}s")

    # Test prompt similar to orchestrator
    prompt = """You are investigating an anomaly: high GC percentage (3.68%) on WebLogic server UIServer_as1_01.

Available tools: search_specialist, invoke_specialists, evaluate_progress, submit_investigation, correlate_and_analyze

Your task: Invoke the appropriate specialists to investigate this issue. Start by invoking wls-health-agent."""

    print(f"Invoking LLM...")
    start = time.time()

    try:
        result = llm_with_tools.invoke(prompt)
        elapsed = time.time() - start
        print(f"Response in {elapsed:.2f}s")
        print(f"Tool calls: {len(result.tool_calls)} - {[tc['name'] for tc in result.tool_calls]}")
        for tc in result.tool_calls:
            print(f"  - {tc['name']}: {tc['args']}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing optic's Garthok provider with ALL 5 orchestrator tools")
    test_garthok_with_all_tools()
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
