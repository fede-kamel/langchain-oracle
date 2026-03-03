#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# ruff: noqa: T201, E501

"""Deep Research Story Generator - Creates narrative from Wikipedia knowledge.

Demonstrates deep agent features:
1. Parallel research queries across multiple topics
2. Cross-referencing and synthesizing diverse information
3. Long-form content generation with Gemini 2.5 Pro
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import oci
import oracledb
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaandceai675euuovyyazlymnglde2xknsq35rni43zzmwdhxxu4v7q",
)
GENAI_REGION = "us-chicago-1"
DB_DSN = "tcps://adb.ca-toronto-1.oraclecloud.com:1522/g4549e8afff78c6_deepresearch_low.adb.oraclecloud.com"
DB_USER = "ADMIN"
DB_PASSWORD = "Research2026Pass#"


def get_clients():
    """Initialize OCI clients."""
    config = oci.config.from_file(profile_name="API_KEY_AUTH")
    config["region"] = GENAI_REGION

    genai_client = GenerativeAiInferenceClient(
        config,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
    )

    db_conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)

    return genai_client, db_conn


def vector_search(conn, genai_client, query: str, top_k: int = 10) -> list[dict]:
    """Perform semantic vector search on Wikipedia dataset."""
    # Generate embedding
    embed_request = EmbedTextDetails(
        inputs=[query],
        serving_mode=OnDemandServingMode(model_id="cohere.embed-english-v3.0"),
        compartment_id=COMPARTMENT_ID,
        input_type="SEARCH_QUERY",
        truncate="END",
    )
    response = genai_client.embed_text(embed_request)
    query_embedding = response.data.embeddings[0]

    cursor = conn.cursor()

    # Search only Wikipedia dataset
    sql = """
        SELECT title, content, dataset, source,
               VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
        FROM VECTOR_DOCUMENTS
        WHERE dataset = 'wikipedia'
        ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
        FETCH FIRST :top_k ROWS ONLY
    """
    params = {"query_vec": json.dumps(query_embedding), "top_k": top_k}

    cursor.execute(sql, params)

    results = []
    for row in cursor:
        content = row[1].read() if hasattr(row[1], "read") else str(row[1])
        results.append(
            {
                "title": row[0],
                "content": content,
                "dataset": row[2],
                "source": row[3],
                "similarity": 1 - row[4],
            }
        )

    cursor.close()
    return results


def research_topic(conn, genai_client, topic: str, query: str) -> dict:
    """Execute a single research query."""
    print(f"  🔍 Researching: {topic}...")
    results = vector_search(conn, genai_client, query, top_k=8)
    return {
        "topic": topic,
        "query": query,
        "results": results,
        "count": len(results),
    }


def parallel_research(conn, genai_client, queries: list[dict]) -> list[dict]:
    """Execute multiple research queries in parallel."""
    print("\n" + "=" * 70)
    print("PARALLEL RESEARCH PHASE - Gathering Wikipedia Knowledge")
    print("=" * 70)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for q in queries:
            future = executor.submit(
                research_topic,
                conn,
                genai_client,
                q["topic"],
                q["query"],
            )
            futures.append(future)

        for future in futures:
            results.append(future.result())

    print(f"\n  ✅ Completed {len(results)} parallel research queries\n")
    return results


def generate_story(genai_client, research_results: list[dict]) -> str:
    """Use Gemini to synthesize a creative story from Wikipedia knowledge."""
    print("=" * 70)
    print("SYNTHESIS PHASE - Gemini Creating Story from Knowledge")
    print("=" * 70)

    # Build context from research
    knowledge_context = ""
    for r in research_results:
        knowledge_context += f"\n\n### KNOWLEDGE AREA: {r['topic']}\n"
        knowledge_context += f"Research Query: {r['query']}\n"
        knowledge_context += f"Found {r['count']} relevant articles:\n\n"
        for i, doc in enumerate(r["results"], 1):
            knowledge_context += (
                f"**Article {i}** (relevance: {doc['similarity']:.2f})\n"
            )
            knowledge_context += f"Title: {doc['title']}\n"
            knowledge_context += f"Content:\n{doc['content'][:2500]}\n\n"

    prompt = f"""You are a master storyteller and creative writer. Using ONLY the factual knowledge
provided below from Wikipedia, craft an engaging, imaginative 5-page short story.

## WIKIPEDIA KNOWLEDGE BASE:
{knowledge_context}

## YOUR TASK:
Create a compelling 5-page short story (approximately 2,500-3,000 words) that:

1. **Weaves together** multiple pieces of knowledge from the research into a coherent narrative
2. **Creates vivid characters** whose journey connects to the historical/scientific facts
3. **Uses accurate details** from the Wikipedia articles to enrich the world-building
4. **Balances education and entertainment** - the reader should learn while being engaged
5. **Has a clear narrative arc** with beginning, rising action, climax, and resolution

## STORY STRUCTURE:

### PAGE 1: THE OPENING
- Introduce the protagonist and setting
- Establish the world using real historical/scientific details from the research
- Hook the reader with an intriguing situation

### PAGE 2: THE JOURNEY BEGINS
- The protagonist encounters a challenge or discovery
- Incorporate factual knowledge naturally into the narrative
- Develop secondary characters

### PAGE 3: RISING ACTION
- Complications arise
- Weave in more knowledge from the research
- Build tension and stakes

### PAGE 4: THE CLIMAX
- The protagonist faces their greatest challenge
- Knowledge from research plays a crucial role
- Dramatic turning point

### PAGE 5: RESOLUTION
- Consequences of the climax unfold
- Character growth revealed
- Satisfying conclusion that ties together the themes

## FORMATTING:
- Use clear page breaks (--- PAGE X ---)
- Include vivid descriptions and dialogue
- Cite the actual Wikipedia knowledge when relevant (naturally woven in)
- Make it feel like a real published short story

Write the complete 5-page story now:"""

    print("  🤖 Gemini crafting story from Wikipedia knowledge...")

    from langchain_oci import ChatOCIGenAI

    llm = ChatOCIGenAI(
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com",
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        model_kwargs={
            "temperature": 0.8,  # Higher for creativity
            "max_tokens": 32000,
        },
    )

    response = llm.invoke(prompt)
    return response.content


def main():
    """Run deep research story generation."""
    print("=" * 70)
    print("DEEP RESEARCH AGENT - STORY GENERATOR")
    print("Oracle 26ai Vector DB + Gemini 2.5 Pro")
    print("=" * 70)

    print("\n## DEEP AGENT FEATURES DEMONSTRATED:")
    print("  1. 🔄 Parallel Research Queries - Multiple topics searched simultaneously")
    print("  2. 🧠 Semantic Vector Search - AI-powered relevance matching")
    print("  3. 📚 Cross-Domain Synthesis - Combining diverse knowledge")
    print("  4. ✍️  Long-Form Generation - Extended creative content")
    print("  5. 🎯 Context-Aware Output - Grounded in real knowledge")

    # Initialize
    print("\nInitializing connections...")
    genai_client, conn = get_clients()
    print("  ✅ Connected to Oracle 26ai Vector Database")
    print("  ✅ Connected to OCI GenAI (Gemini)")

    # Define research queries to gather diverse Wikipedia knowledge
    research_queries = [
        {
            "topic": "Ancient Civilizations",
            "query": "ancient civilization history empire culture society achievements",
        },
        {
            "topic": "Scientific Discovery",
            "query": "scientific discovery invention breakthrough research laboratory",
        },
        {
            "topic": "Natural Wonders",
            "query": "nature landscape mountain ocean forest wildlife ecosystem",
        },
        {
            "topic": "Historical Figures",
            "query": "famous person leader inventor scientist explorer biography",
        },
        {
            "topic": "Technology & Innovation",
            "query": "technology innovation engineering machine computer development",
        },
        {
            "topic": "Art & Culture",
            "query": "art culture music literature painting sculpture creative",
        },
    ]

    # Execute parallel research
    research_results = parallel_research(conn, genai_client, research_queries)

    # Display research summary
    print("=" * 70)
    print("KNOWLEDGE GATHERED")
    print("=" * 70)
    total_docs = 0
    for r in research_results:
        print(f"\n📚 {r['topic']}")
        print(f"   Found: {r['count']} Wikipedia articles")
        if r["results"]:
            top = r["results"][0]
            print(f"   Top: {top['title'][:60]}... (sim: {top['similarity']:.2f})")
        total_docs += r["count"]

    print(f"\n  Total Wikipedia articles gathered: {total_docs}")

    # Generate story
    story = generate_story(genai_client, research_results)

    # Output
    print("\n" + "=" * 70)
    print("THE STORY")
    print("=" * 70)
    print(story)

    # Save to file
    output_path = "/Users/federico.kamelhar/Projects/langchain-oracle/libs/oci/wiki_story_output.md"
    with open(output_path, "w") as f:
        f.write("# Deep Research Agent - Story from Wikipedia Knowledge\n\n")
        f.write("## Agent Features Demonstrated\n")
        f.write("1. **Parallel Research Queries** - 6 topics searched simultaneously\n")
        f.write("2. **Semantic Vector Search** - AI-powered relevance matching\n")
        f.write("3. **Cross-Domain Synthesis** - Combining diverse knowledge\n")
        f.write("4. **Long-Form Generation** - Extended creative content\n")
        f.write(
            "5. **Context-Aware Output** - Grounded in real Wikipedia knowledge\n\n"
        )
        f.write("## Research Statistics\n")
        f.write(f"- Total research queries: {len(research_queries)}\n")
        f.write(f"- Total Wikipedia articles analyzed: {total_docs}\n")
        f.write("- Synthesis model: Gemini 2.5 Pro\n\n")
        f.write("---\n\n")
        f.write(story)

    print(f"\n  📄 Story saved to: {output_path}")

    print("\n" + "=" * 70)
    print("RESEARCH STATISTICS")
    print("=" * 70)
    print(f"  Total research queries: {len(research_queries)}")
    print(f"  Total Wikipedia articles analyzed: {total_docs}")
    print("  Synthesis model: Gemini 2.5 Pro")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
