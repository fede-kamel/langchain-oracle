# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Sample research datasets for testing deep agents.

These datasets are inspired by:
- DeepResearch Bench (https://github.com/Ayanami0730/deep_research_bench)
- LangChain evaluation patterns

Usage:
    from tests.integration_tests.agents.research_datasets import (
        BASIC_RESEARCH_TASKS,
        ANALYSIS_TASKS,
        COMPARISON_TASKS,
    )

    for task in BASIC_RESEARCH_TASKS:
        result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})
        # Evaluate result against task["expected_topics"]
"""

from typing import TypedDict


class ResearchTask(TypedDict):
    """A research task for evaluation."""

    id: str
    query: str
    expected_topics: list[str]
    difficulty: str  # "easy", "medium", "hard"
    domain: str
    description: str


# Basic research tasks - single topic, straightforward queries
BASIC_RESEARCH_TASKS: list[ResearchTask] = [
    {
        "id": "basic_ai_1",
        "query": "What is machine learning and how does it work?",
        "expected_topics": ["learning", "data", "algorithm", "model"],
        "difficulty": "easy",
        "domain": "AI",
        "description": "Basic explanation of machine learning concepts",
    },
    {
        "id": "basic_cloud_1",
        "query": "Explain cloud computing and its main benefits.",
        "expected_topics": ["cloud", "service", "scalability", "cost"],
        "difficulty": "easy",
        "domain": "Cloud",
        "description": "Introduction to cloud computing",
    },
    {
        "id": "basic_security_1",
        "query": "What is cybersecurity and why is it important?",
        "expected_topics": ["security", "threat", "protection", "data"],
        "difficulty": "easy",
        "domain": "Security",
        "description": "Cybersecurity fundamentals",
    },
    {
        "id": "basic_quantum_1",
        "query": "What is quantum computing?",
        "expected_topics": ["quantum", "qubit", "superposition"],
        "difficulty": "easy",
        "domain": "Quantum",
        "description": "Introduction to quantum computing",
    },
]

# Analysis tasks - require synthesis of multiple information sources
ANALYSIS_TASKS: list[ResearchTask] = [
    {
        "id": "analysis_ai_trends",
        "query": (
            "Analyze the current state of AI development. "
            "What are the major trends and challenges?"
        ),
        "expected_topics": ["trend", "model", "challenge", "development"],
        "difficulty": "medium",
        "domain": "AI",
        "description": "Analysis of AI industry trends",
    },
    {
        "id": "analysis_cloud_market",
        "query": (
            "Analyze the cloud computing market. "
            "Who are the major players and what's the market outlook?"
        ),
        "expected_topics": ["market", "aws", "azure", "growth"],
        "difficulty": "medium",
        "domain": "Cloud",
        "description": "Cloud market analysis",
    },
    {
        "id": "analysis_ai_safety",
        "query": (
            "Analyze the current state of AI safety research. "
            "What are the key research areas and challenges?"
        ),
        "expected_topics": ["safety", "alignment", "research", "risk"],
        "difficulty": "medium",
        "domain": "AI Safety",
        "description": "AI safety research analysis",
    },
    {
        "id": "analysis_enterprise_ai",
        "query": (
            "How are enterprises adopting AI? "
            "What are the common use cases and challenges?"
        ),
        "expected_topics": ["enterprise", "adoption", "use case", "challenge"],
        "difficulty": "medium",
        "domain": "Enterprise",
        "description": "Enterprise AI adoption analysis",
    },
]

# Comparison tasks - require comparing multiple concepts/technologies
COMPARISON_TASKS: list[ResearchTask] = [
    {
        "id": "compare_cloud_providers",
        "query": (
            "Compare AWS, Azure, and Oracle Cloud. "
            "What are their strengths and target markets?"
        ),
        "expected_topics": ["aws", "azure", "oracle", "strength"],
        "difficulty": "hard",
        "domain": "Cloud",
        "description": "Cloud provider comparison",
    },
    {
        "id": "compare_ml_approaches",
        "query": (
            "Compare supervised and unsupervised machine learning. "
            "When should each approach be used?"
        ),
        "expected_topics": ["supervised", "unsupervised", "learning", "use"],
        "difficulty": "medium",
        "domain": "AI",
        "description": "ML approach comparison",
    },
    {
        "id": "compare_quantum_classical",
        "query": (
            "Compare quantum and classical computers for optimization. "
            "What are the advantages of each?"
        ),
        "expected_topics": ["quantum", "classical", "optimization", "advantage"],
        "difficulty": "hard",
        "domain": "Quantum",
        "description": "Quantum vs classical computing comparison",
    },
]

# Multi-step research tasks - require planning and multiple tool calls
MULTI_STEP_TASKS: list[ResearchTask] = [
    {
        "id": "multi_tech_landscape",
        "query": (
            "Research the current technology landscape for enterprise AI adoption. "
            "Include market statistics, key trends, and recommendations."
        ),
        "expected_topics": ["enterprise", "ai", "market", "trend", "recommend"],
        "difficulty": "hard",
        "domain": "Enterprise",
        "description": "Comprehensive enterprise AI landscape research",
    },
    {
        "id": "multi_cloud_strategy",
        "query": (
            "Research best practices for cloud migration. "
            "Include current trends, common challenges, and success factors."
        ),
        "expected_topics": ["migration", "cloud", "best practice", "challenge"],
        "difficulty": "hard",
        "domain": "Cloud",
        "description": "Cloud migration strategy research",
    },
    {
        "id": "multi_security_assessment",
        "query": (
            "Research current cybersecurity threats and defenses. "
            "Include threat landscape, defense strategies, and emerging technologies."
        ),
        "expected_topics": ["threat", "security", "defense", "technology"],
        "difficulty": "hard",
        "domain": "Security",
        "description": "Cybersecurity landscape research",
    },
]

# All tasks combined
ALL_RESEARCH_TASKS: list[ResearchTask] = (
    BASIC_RESEARCH_TASKS + ANALYSIS_TASKS + COMPARISON_TASKS + MULTI_STEP_TASKS
)


def get_tasks_by_difficulty(difficulty: str) -> list[ResearchTask]:
    """Get tasks filtered by difficulty level."""
    return [t for t in ALL_RESEARCH_TASKS if t["difficulty"] == difficulty]


def get_tasks_by_domain(domain: str) -> list[ResearchTask]:
    """Get tasks filtered by domain."""
    return [t for t in ALL_RESEARCH_TASKS if t["domain"].lower() == domain.lower()]


def evaluate_response(response: str, task: ResearchTask) -> dict:
    """Evaluate a response against expected topics.

    Args:
        response: The agent's response text
        task: The research task with expected topics

    Returns:
        dict with evaluation metrics
    """
    response_lower = response.lower()
    expected = task["expected_topics"]
    found = [topic for topic in expected if topic in response_lower]

    return {
        "task_id": task["id"],
        "topics_expected": len(expected),
        "topics_found": len(found),
        "coverage": len(found) / len(expected) if expected else 0,
        "found_topics": found,
        "missing_topics": [t for t in expected if t not in response_lower],
        "response_length": len(response),
        "passed": len(found) >= 1,  # At least one topic mentioned
    }
