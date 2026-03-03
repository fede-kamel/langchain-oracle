#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201
"""Test async support for Deep Research Agent with Gemini 2.5 Pro."""

import asyncio
import os
import time

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)
GENAI_REGION = "us-chicago-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com"


def test_sync_invoke():
    """Test synchronous invocation."""
    print("\n" + "=" * 60)
    print("TEST 1: Synchronous Invocation (invoke)")
    print("=" * 60)

    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from langchain_oci.agents import create_deep_research_agent

    @tool
    def get_current_time() -> str:
        """Get the current UTC time."""
        import datetime

        return datetime.datetime.now(datetime.UTC).isoformat()

    print("\n  Creating agent with google.gemini-2.5-pro...")
    agent = create_deep_research_agent(
        tools=[get_current_time],
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        temperature=0.3,
        max_tokens=1024,
    )
    print("  Agent created!")

    query = "What time is it? Use the tool to check."
    print(f"\n  Query: '{query}'")

    start = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    elapsed = time.time() - start

    response = result["messages"][-1].content
    print(f"\n  Response: {response[:200]}...")
    print(f"  Time: {elapsed:.2f}s")
    print("\n  [PASS] Sync invocation works")
    return True


async def test_async_invoke():
    """Test asynchronous invocation."""
    print("\n" + "=" * 60)
    print("TEST 2: Asynchronous Invocation (ainvoke)")
    print("=" * 60)

    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from langchain_oci.agents import create_deep_research_agent

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression safely."""
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            try:
                return str(eval(expression))  # noqa: S307
            except Exception as e:
                return f"Error: {e}"
        return "Error: Invalid expression"

    print("\n  Creating agent with google.gemini-2.5-pro...")
    agent = create_deep_research_agent(
        tools=[calculate],
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        temperature=0.3,
        max_tokens=1024,
    )
    print("  Agent created!")

    query = "What is 42 * 17 + 123? Use the calculator tool."
    print(f"\n  Query: '{query}'")

    start = time.time()
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        elapsed = time.time() - start

        response = result["messages"][-1].content
        print(f"\n  Response: {response[:200]}...")
        print(f"  Time: {elapsed:.2f}s")
        print("\n  [PASS] Async invocation works")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] Async error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        import traceback

        traceback.print_exc()
        return False


async def test_async_stream():
    """Test asynchronous streaming."""
    print("\n" + "=" * 60)
    print("TEST 3: Asynchronous Streaming (astream)")
    print("=" * 60)

    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from langchain_oci.agents import create_deep_research_agent

    @tool
    def search(query: str) -> str:
        """Search for information (mock)."""
        return f"Mock results for: {query}"

    print("\n  Creating agent with google.gemini-2.5-pro...")
    agent = create_deep_research_agent(
        tools=[search],
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        temperature=0.3,
        max_tokens=512,
    )
    print("  Agent created!")

    query = "Tell me briefly about AI agents."
    print(f"\n  Query: '{query}'")
    print("\n  Streaming response:")
    print("  " + "-" * 50)

    start = time.time()
    try:
        chunks_received = 0
        async for chunk in agent.astream({"messages": [HumanMessage(content=query)]}):
            chunks_received += 1
            # Print chunk type for debugging
            if chunks_received <= 3:
                chunk_keys = list(chunk.keys()) if isinstance(chunk, dict) else type(chunk).__name__
                print(f"  Chunk {chunks_received}: {chunk_keys}")

        elapsed = time.time() - start
        print("  " + "-" * 50)
        print(f"  Total chunks: {chunks_received}")
        print(f"  Time: {elapsed:.2f}s")
        print("\n  [PASS] Async streaming works")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] Streaming error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        import traceback

        traceback.print_exc()
        return False


async def test_concurrent_requests():
    """Test concurrent async requests."""
    print("\n" + "=" * 60)
    print("TEST 4: Concurrent Async Requests")
    print("=" * 60)

    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from langchain_oci.agents import create_deep_research_agent

    @tool
    def echo(message: str) -> str:
        """Echo the message back."""
        return f"Echo: {message}"

    print("\n  Creating agent with google.gemini-2.5-pro...")
    agent = create_deep_research_agent(
        tools=[echo],
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        temperature=0.3,
        max_tokens=256,
    )
    print("  Agent created!")

    queries = [
        "Echo 'Hello' using the tool",
        "Echo 'World' using the tool",
        "Echo 'Async' using the tool",
    ]

    print(f"\n  Running {len(queries)} concurrent requests...")

    start = time.time()
    try:
        tasks = [
            agent.ainvoke({"messages": [HumanMessage(content=q)]}) for q in queries
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print("\n  Results:")
        for i, result in enumerate(results):
            response = result["messages"][-1].content[:80]
            print(f"    {i+1}. {response}...")

        print(f"\n  Total time for {len(queries)} concurrent: {elapsed:.2f}s")
        print(f"  Avg per request: {elapsed/len(queries):.2f}s")
        print("\n  [PASS] Concurrent async requests work")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] Concurrent error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all async tests."""
    print("=" * 60)
    print("DEEP RESEARCH AGENT - ASYNC SUPPORT TEST")
    print("=" * 60)
    print(f"Model: google.gemini-2.5-pro")
    print(f"Region: {GENAI_REGION}")

    results = []

    # Test 1: Sync
    try:
        results.append(("Sync invoke", test_sync_invoke()))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("Sync invoke", False))

    # Test 2: Async invoke
    results.append(("Async ainvoke", await test_async_invoke()))

    # Test 3: Async stream
    results.append(("Async astream", await test_async_stream()))

    # Test 4: Concurrent
    results.append(("Concurrent requests", await test_concurrent_requests()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")
        if result:
            passed += 1

    print(f"\n  Total: {passed}/{len(results)} tests passed")
    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    import sys

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
