# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Deep Research Agent - deepagents-based research agent with OCI GenAI."""

from __future__ import annotations

import copy
import re
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from langchain.agents import create_agent
from langchain_oci.agents.common import OCIConfig, filter_none, merge_model_kwargs
from langchain_oci.agents.datastores import VectorDataStore, create_datastore_tools
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


DATASTORE_RESEARCH_PROMPT = (
    "When datastores are available, treat them as the primary evidence source. "
    "Users may refer to the backend technology (for example OpenSearch or ADB) "
    "instead of the configured datastore name; map those requests to the relevant "
    "available datastore rather than refusing the task. Prefer semantic search "
    "and get_document to gather evidence, and use stats only when you need store "
    "inventory or size information. For corpus-summary or research-memo tasks, "
    "do one broad search first, retrieve only the most representative documents, "
    "and use at most one follow-up keyword search to fill obvious gaps. Do not "
    "repeat redundant searches once the main themes are covered. If the user asks "
    "for a long memo, expand the synthesis from the evidence you already have "
    "instead of continuing to browse the same small corpus. After you have enough "
    "evidence, write the final research memo directly and stop. Do not end with "
    "an empty response."
)


def _install_output_schema_fallback(compiled: Any) -> Any:
    """Install a permissive output schema fallback for upstream schema bugs.

    Deep agents currently expose state annotations that can fail schema creation
    in langgraph/pydantic when `output_schema` is accessed. Execution still works,
    but schema introspection crashes. We preserve the normal behavior and only
    fall back to an `Any`-typed shape when upstream raises.
    """
    original_get_output_schema = getattr(compiled, "get_output_schema", None)
    if original_get_output_schema is None or not callable(original_get_output_schema):
        return compiled

    def safe_get_output_schema(self: Any, config: Any = None) -> type[BaseModel]:
        try:
            return original_get_output_schema(config)
        except Exception as ex:
            if ex.__class__.__name__ not in {
                "PydanticForbiddenQualifier",
                "TypeError",
            }:
                raise

            field_definitions: dict[str, tuple[Any, Any]] = {}
            output_channels = getattr(self, "output_channels", ())
            if isinstance(output_channels, str):
                field_definitions[output_channels] = (Any, None)
            else:
                for channel in output_channels or ():
                    field_definitions[channel] = (Any, None)

            if not field_definitions:
                field_definitions["output"] = (Any, None)

            model_name_getter = getattr(self, "get_name", None)
            model_name = (
                model_name_getter("Output")
                if callable(model_name_getter)
                else "DeepResearchOutput"
            )
            return create_model(model_name, **field_definitions)

    setattr(compiled, "get_output_schema", MethodType(safe_get_output_schema, compiled))
    return compiled


def _extract_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
    return []


def _terminal_message_is_empty(result: Any) -> bool:
    messages = _extract_messages(result)
    if not messages:
        return False
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return False
    return not str(last_message.content or "").strip()


def _build_recovery_prompt(messages: list[Any]) -> str | None:
    user_requests: list[str] = []
    evidence_blocks: list[str] = []

    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(message.content or "").strip()
            if content:
                user_requests.append(content)
        elif isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content:
                evidence_blocks.append(content)

    if not user_requests or not evidence_blocks:
        return None

    evidence_text = "\n\n".join(evidence_blocks)
    # Keep the recovery prompt bounded even if the agent explored heavily.
    if len(evidence_text) > 24000:
        evidence_text = evidence_text[:24000]

    request_text = "\n".join(user_requests)
    return (
        "The prior Deep Research run collected datastore evidence but returned an "
        "empty final answer. Write the final answer now using only the evidence "
        "below. Do not call tools. Do not ask follow-up questions. If the user "
        "asked for Markdown, return Markdown.\n\n"
        f"Original user request:\n{request_text}\n\n"
        f"Collected evidence:\n{evidence_text}"
    )


def _recover_empty_terminal_message(result: Any, llm: ChatOCIGenAI) -> Any:
    if not _terminal_message_is_empty(result):
        return result

    messages = _extract_messages(result)
    prompt = _build_recovery_prompt(messages)
    if not prompt:
        return result

    recovery = llm.invoke([HumanMessage(content=prompt)])
    if not str(recovery.content or "").strip():
        return result

    recovered_messages = list(messages[:-1]) + [recovery]
    recovered_result = dict(result)
    recovered_result["messages"] = recovered_messages
    return recovered_result


def _rewrite_backend_aliases_in_input(
    input: Any,
    datastores: dict[str, VectorDataStore] | None,
) -> Any:
    if not datastores or len(datastores) != 1 or not isinstance(input, dict):
        return input

    messages = input.get("messages")
    if not isinstance(messages, list):
        return input

    store_name, store = next(iter(datastores.items()))
    backend_name = getattr(store, "name", "").strip().lower()
    if backend_name not in {"opensearch", "adb"}:
        return input

    alias_patterns = [
        re.compile(rf"\bthe\s+{re.escape(backend_name)}\s+datastore\b", re.IGNORECASE),
        re.compile(rf"\b{re.escape(backend_name)}\s+datastore\b", re.IGNORECASE),
    ]
    rewritten = False
    updated_messages: list[Any] = []

    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(message.content or "")
            new_content = content
            for pattern in alias_patterns:
                new_content = pattern.sub(f"{store_name} datastore", new_content)
            if new_content != content:
                updated_messages.append(HumanMessage(content=new_content))
                rewritten = True
            else:
                updated_messages.append(message)
        elif isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", ""))
            new_content = content
            for pattern in alias_patterns:
                new_content = pattern.sub(f"{store_name} datastore", new_content)
            if new_content != content:
                updated = dict(message)
                updated["content"] = new_content
                updated_messages.append(updated)
                rewritten = True
            else:
                updated_messages.append(message)
        else:
            updated_messages.append(message)

    if not rewritten:
        return input

    updated_input = copy.deepcopy(input)
    updated_input["messages"] = updated_messages
    return updated_input


async def _arecover_empty_terminal_message(result: Any, llm: ChatOCIGenAI) -> Any:
    if not _terminal_message_is_empty(result):
        return result

    messages = _extract_messages(result)
    prompt = _build_recovery_prompt(messages)
    if not prompt:
        return result

    recovery = await llm.ainvoke([HumanMessage(content=prompt)])
    if not str(recovery.content or "").strip():
        return result

    recovered_messages = list(messages[:-1]) + [recovery]
    recovered_result = dict(result)
    recovered_result["messages"] = recovered_messages
    return recovered_result


def _install_empty_response_recovery(compiled: Any, llm: ChatOCIGenAI) -> Any:
    original_invoke = getattr(compiled, "invoke", None)
    if callable(original_invoke):

        def safe_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            result = original_invoke(input, config=config, **kwargs)
            return _recover_empty_terminal_message(result, llm)

        setattr(compiled, "invoke", MethodType(safe_invoke, compiled))

    original_ainvoke = getattr(compiled, "ainvoke", None)
    if callable(original_ainvoke):

        async def safe_ainvoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            result = await original_ainvoke(input, config=config, **kwargs)
            return await _arecover_empty_terminal_message(result, llm)

        setattr(compiled, "ainvoke", MethodType(safe_ainvoke, compiled))

    return compiled


def _install_backend_alias_rewrite(
    compiled: Any,
    datastores: dict[str, VectorDataStore] | None,
) -> Any:
    original_invoke = getattr(compiled, "invoke", None)
    if callable(original_invoke):

        def alias_safe_invoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            rewritten_input = _rewrite_backend_aliases_in_input(input, datastores)
            return original_invoke(rewritten_input, config=config, **kwargs)

        setattr(compiled, "invoke", MethodType(alias_safe_invoke, compiled))

    original_ainvoke = getattr(compiled, "ainvoke", None)
    if callable(original_ainvoke):

        async def alias_safe_ainvoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            rewritten_input = _rewrite_backend_aliases_in_input(input, datastores)
            return await original_ainvoke(rewritten_input, config=config, **kwargs)

        setattr(compiled, "ainvoke", MethodType(alias_safe_ainvoke, compiled))

    return compiled


def _compose_system_prompt(
    system_prompt: str | None,
    *,
    has_datastores: bool,
) -> str | None:
    if not has_datastores:
        return system_prompt
    if system_prompt:
        return f"{DATASTORE_RESEARCH_PROMPT}\n\n{system_prompt}"
    return DATASTORE_RESEARCH_PROMPT


def _should_use_lightweight_datastore_agent(
    *,
    datastores: dict[str, VectorDataStore] | None,
    subagents: list[Any] | None,
    skills: list[str] | None,
    memory: list[str] | None,
) -> bool:
    return bool(datastores) and not subagents and not skills and not memory


def create_deep_research_agent(
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    *,
    # Datastores - if provided, auto-routing search is enabled
    datastores: dict[str, VectorDataStore] | None = None,
    default_datastore: str | None = None,
    default_store: str | None = None,  # Alias for default_datastore
    embedding_model: Any = None,
    top_k: int = 5,
    # OCI-specific options
    model_id: str = "google.gemini-2.5-pro",
    compartment_id: str | None = None,
    service_endpoint: str | None = None,
    auth_type: str | OCIAuthType = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    # Deep agent options
    system_prompt: str | None = None,
    subagents: list[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    middleware: Sequence[Any] | None = None,
    # LangGraph options
    checkpointer: Any = None,
    store: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    # Model kwargs
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,  # noqa: ARG001 - Intentionally ignored
    **model_kwargs: Any,
) -> "CompiledStateGraph":
    """Create a Deep Research Agent using OCI GenAI and deepagents.

    This agent is designed for multi-step research tasks that require:
    - Searching multiple data sources (OpenSearch, ADB)
    - Planning and reflection
    - Synthesizing information into reports

    Args:
        tools: Custom tools for the agent.
        datastores: Dict of vector datastores for auto-routing search.
        default_datastore: Fallback datastore if routing is inconclusive.
        default_store: Alias for default_datastore.
        embedding_model: Custom embedding model for datastores.
        top_k: Number of search results to return.
        model_id: OCI model identifier (Gemini models recommended).
        compartment_id: OCI compartment OCID.
        service_endpoint: OCI GenAI service endpoint.
        auth_type: OCI authentication type.
        auth_profile: OCI config profile name.
        auth_file_location: Path to OCI config file.
        system_prompt: Custom system prompt for the agent.
        subagents: List of subagents for delegation.
        skills: List of skill names to enable.
        memory: List of memory namespaces.
        middleware: Custom middleware. Pass empty list to disable defaults.
        checkpointer: LangGraph checkpointer for persistence/memory.
        store: LangGraph store for long-term memory.
        interrupt_before: Tools to pause before for human-in-loop.
        interrupt_after: Tools to pause after for human-in-loop.
        debug: Enable debug mode.
        name: Name for the agent.
        temperature: Model temperature.
        max_tokens: Maximum output tokens (e.g., 65536 for Gemini 2.5 Pro).
        max_input_tokens: Ignored. Input limits are model-determined.
        **model_kwargs: Additional model kwargs.

    Returns:
        CompiledStateGraph: A compiled deep research agent.

    Example:
        >>> from langchain_oci.agents.deep_research import OpenSearch, ADB
        >>>
        >>> agent = create_deep_research_agent(
        ...     datastores={
        ...         "docs": OpenSearch(
        ...             endpoint="https://opensearch:9200",
        ...             index_name="company-docs",
        ...             datastore_description="internal documentation, policies",
        ...         ),
        ...         "sales": ADB(
        ...             dsn="mydb_low",
        ...             user="ADMIN",
        ...             password="...",
        ...             datastore_description="sales data, revenue, customers",
        ...         ),
        ...     },
        ...     compartment_id="ocid1.compartment...",
        ... )
    """
    # Resolve OCI configuration
    oci_config = OCIConfig.resolve(
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
    )

    # Build tools list
    all_tools: list[BaseTool | Callable[..., Any]] = []

    has_datastores = bool(datastores)

    if datastores:
        datastore_tools = create_datastore_tools(
            stores=datastores,
            default_store=default_store or default_datastore,
            embedding_model=embedding_model,
            compartment_id=oci_config.compartment_id,
            service_endpoint=oci_config.service_endpoint,
            auth_type=oci_config.auth_type,
            auth_profile=oci_config.auth_profile,
            top_k=top_k,
        )
        all_tools.extend(datastore_tools)

    if tools:
        all_tools.extend(tools)

    effective_system_prompt = _compose_system_prompt(
        system_prompt,
        has_datastores=has_datastores,
    )

    # Create OCI chat model
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=oci_config.compartment_id,
        service_endpoint=oci_config.service_endpoint,
        auth_type=oci_config.auth_type,
        auth_profile=oci_config.auth_profile,
        auth_file_location=oci_config.auth_file_location,
        model_kwargs=merge_model_kwargs(
            model_kwargs,
            temperature,
            max_tokens,
            model_id=model_id,
        ),
    )

    if _should_use_lightweight_datastore_agent(
        datastores=datastores,
        subagents=subagents,
        skills=skills,
        memory=memory,
    ):
        compiled = create_agent(
            llm,
            tools=all_tools,
            **filter_none(
                system_prompt=effective_system_prompt,
                middleware=tuple(middleware or ()),
                checkpointer=checkpointer,
                store=store,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                name=name,
            ),
            debug=debug,
        )
    else:
        try:
            from deepagents import create_deep_agent
        except ImportError as ex:
            raise ImportError(
                "deepagents required. Install with: pip install deepagents"
            ) from ex

        # Build agent kwargs - only include non-None values
        agent_kwargs = {
            "model": llm,
            "tools": all_tools,
            **filter_none(
                system_prompt=effective_system_prompt,
                subagents=subagents,
                skills=skills,
                memory=memory,
                middleware=middleware,
                checkpointer=checkpointer,
                store=store,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                name=name,
            ),
        }

        # debug=False is meaningful, so handle separately
        if debug:
            agent_kwargs["debug"] = True

        compiled = create_deep_agent(**agent_kwargs)
    compiled = _install_backend_alias_rewrite(compiled, datastores)
    compiled = _install_output_schema_fallback(compiled)
    compiled = _install_empty_response_recovery(compiled, llm)
    # Expose the underlying OCI chat model for explicit cleanup in long-lived
    # processes (and in our integration tests). This avoids aiohttp
    # "Unclosed client session" warnings when async pooling is used.
    setattr(compiled, "_oci_llm", llm)
    return compiled
