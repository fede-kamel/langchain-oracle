# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
# ruff: noqa: T201

"""
Example: Creating a Deep Agent with OCI Generative AI

This example demonstrates how to create a Deep Agent using OCI's
generative AI models. Deep Agents have built-in capabilities for
planning (write_todos), file operations, and subagent spawning.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install langchain-oci deepagents
   ```

2. Set up OCI authentication:
   ```bash
   oci session authenticate
   ```

3. Set environment variables:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   export OCI_REGION="us-chicago-1"  # or your region
   ```

## Running the Example

```bash
cd libs/oci
python examples/agents/deep_agent_example.py
```
"""

import os

from langchain_core.tools import tool

from langchain_oci import create_oci_deep_agent


# Define custom tools for your use case
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information."""
    # In a real application, this would query your knowledge base
    knowledge = {
        "sales": "Q4 2025 sales reached $2.1M, up 15% YoY.",
        "customers": "Total active customers: 1,250. Top segment: Enterprise.",
        "products": "Top products: Cloud Services (45%), Analytics (30%).",
        "trends": "Key trends: AI adoption up 40%, cloud migration accelerating.",
    }
    for topic, info in knowledge.items():
        if topic in query.lower():
            return f"Knowledge Base Result:\n{info}"
    return f"No results found for '{query}'. Try: sales, customers, products, trends"


@tool
def query_database(sql_description: str) -> str:
    """Query the database based on a natural language description."""
    # In a real application, this would convert to SQL and query your DB
    mock_results = {
        "revenue": (
            "Total Revenue: $8.4M\nBy Quarter: Q1=$1.8M, Q2=$2.1M, Q3=$2.4M, Q4=$2.1M"
        ),
        "users": "Active Users: 15,420\nNew This Month: 1,250\nChurn: 2.3%",
        "performance": "System Uptime: 99.97%\nAvg Response Time: 145ms",
    }
    for key, result in mock_results.items():
        if key in sql_description.lower():
            return f"Database Query Result:\n{result}"
    return "Query executed. No matching data found."


@tool
def get_market_analysis(topic: str) -> str:
    """Get market analysis for a given topic."""
    # In a real application, this might call an external API
    analyses = {
        "cloud": "Cloud market growing at 22% CAGR. Key: AWS, Azure, OCI, GCP.",
        "ai": "AI/ML market to reach $500B by 2027. Focus: GenAI, AutoML.",
        "security": "Cybersecurity spending up 12%. Zero-trust adoption increasing.",
    }
    for key, analysis in analyses.items():
        if key in topic.lower():
            return f"Market Analysis:\n{analysis}"
    return f"No market analysis available for '{topic}'."


def main() -> None:
    """Run the example deep agent."""
    # Get configuration from environment
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        raise ValueError("Please set OCI_COMPARTMENT_ID environment variable")

    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    print("Creating OCI Deep Agent...")
    print("  Model: meta.llama-4-scout-17b-16e-instruct")
    print(f"  Region: {region}")
    print("  Custom Tools: search_knowledge_base, query_database, get_market_analysis")
    print("  Built-in: write_todos (planning), file ops, subagent spawning")
    print()

    # Create the deep agent with custom tools
    agent = create_oci_deep_agent(
        tools=[search_knowledge_base, query_database, get_market_analysis],
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="SECURITY_TOKEN",
        system_prompt="""You are a research analyst assistant.
You help users research topics by:
1. First planning your approach using write_todos
2. Gathering information from the knowledge base and database
3. Getting market analysis when relevant
4. Synthesizing findings into clear insights

Be thorough but concise. Always cite your sources.""",
        temperature=0.3,
        max_tokens=2048,
    )

    # Example research task
    research_query = """
    Research our company's Q4 performance and provide insights on:
    1. Sales performance
    2. Customer trends
    3. Market position in cloud services
    """

    print("=" * 60)
    print("Research Task:")
    print(research_query.strip())
    print("=" * 60)
    print()
    print("Running deep agent (this may involve multiple steps)...")
    print()

    result = agent.invoke({"messages": [{"role": "user", "content": research_query}]})

    # Print the final response
    print("=" * 60)
    print("Research Results:")
    print("=" * 60)

    final_message = result["messages"][-1]
    print(final_message.content)

    print()
    print("=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
