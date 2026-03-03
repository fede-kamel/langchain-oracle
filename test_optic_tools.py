"""Test langchain-oci with optic-like tool schemas.

Mimics the exact tool structure optic uses to find what causes hangs.
"""

import time
from typing import Any
from pydantic import BaseModel, Field
from langchain_oci import ChatOCIGenAI
from langchain_core.tools import StructuredTool


# ============================================================================
# Tool 1: search_specialist (simple)
# ============================================================================
class SearchSpecialistInput(BaseModel):
    """Input for search_specialist."""
    query: str = Field(..., description="Search query for specialists")
    max_results: int = Field(default=5, description="Max results to return")


def search_specialist_fn(query: str, max_results: int = 5) -> str:
    return f"Found specialists for: {query}"


# ============================================================================
# Tool 2: invoke_specialists (complex - nested list + dict[str, Any])
# ============================================================================
class SpecialistInvocation(BaseModel):
    """Single specialist invocation request."""
    specialist_id: str = Field(..., description="The agent_id of the specialist")
    focus_area: str = Field(..., description="What aspect to focus on")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class InvokeSpecialistsInput(BaseModel):
    """Input for invoke_specialists."""
    invocations: list[SpecialistInvocation] = Field(
        ..., min_length=1, max_length=4, description="List of specialists to invoke"
    )
    timeout_seconds: int = Field(default=300, ge=30, le=600, description="Timeout")


def invoke_specialists_fn(invocations: list[SpecialistInvocation], timeout_seconds: int = 300) -> str:
    return f"Invoked {len(invocations)} specialists"


# ============================================================================
# Tool 3: evaluate_progress (complex - multiple nested types)
# ============================================================================
class EvaluateProgressInput(BaseModel):
    """Input for evaluate_progress."""
    force_complete: bool = Field(default=False, description="Force completion")


def evaluate_progress_fn(force_complete: bool = False) -> str:
    return "Progress evaluated"


# ============================================================================
# Tool 4: submit_investigation (complex - many fields)
# ============================================================================
class SubmitInvestigationInput(BaseModel):
    """Input for submit_investigation."""
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
# Test runner
# ============================================================================
def create_tools() -> list[StructuredTool]:
    """Create optic-like tools."""
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
    ]


def test_with_timeout(tools: list, prompt: str, timeout: int = 60):
    """Test LLM with tools and timeout."""
    print(f"\n{'='*60}")
    print(f"Testing with {len(tools)} tools")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}")

    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaah7ixt2oanvvualoahejm63r66c3pse5u4nd4gzviax7eeeqhrysq",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        model_kwargs={"temperature": 0.2, "max_tokens": 1000},
    )

    print("Binding tools...")
    start = time.time()
    llm_with_tools = llm.bind_tools(tools)
    bind_time = time.time() - start
    print(f"Tools bound in {bind_time:.2f}s")

    print("Invoking LLM...")
    start = time.time()
    try:
        import asyncio
        result = asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: llm_with_tools.invoke(prompt)
            ),
            timeout=timeout
        )
        # For sync call:
        result = llm_with_tools.invoke(prompt)
        elapsed = time.time() - start
        print(f"Response time: {elapsed:.2f}s")
        print(f"Tool calls: {len(result.tool_calls)} - {[tc['name'] for tc in result.tool_calls]}")
        print(f"Content: {result.content[:200] if result.content else '(empty)'}")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {type(e).__name__}: {str(e)[:200]}")
        return False, elapsed


if __name__ == "__main__":
    print("Testing langchain-oci + Llama 4 Maverick with optic-like tools")
    print(f"Model: meta.llama-4-maverick-17b-128e-instruct-fp8")

    tools = create_tools()

    # Test 1: Simple prompt asking to search
    test_with_timeout(
        tools,
        "Search for specialists that can investigate WebLogic memory issues",
    )

    # Test 2: Prompt asking to invoke specialists
    test_with_timeout(
        tools,
        "Invoke the wls-health-agent specialist to investigate high GC percentage on UIServer_as1_01",
    )

    # Test 3: Complex orchestrator-like prompt
    test_with_timeout(
        tools,
        """You are investigating an anomaly: high GC percentage (3.68%) on WebLogic server UIServer_as1_01.

Available tools: search_specialist, invoke_specialists, evaluate_progress, submit_investigation

First, invoke the appropriate specialists to investigate this issue.""",
    )

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
