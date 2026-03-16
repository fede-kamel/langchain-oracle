"""Compatibility patches for upstream LangChain/LangGraph edge cases."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model


def _build_loose_schema(
    model_name: str,
    field_definitions: dict[str, Any] | None = None,
    root: Any | None = None,
) -> type[BaseModel]:
    """Build a permissive model when upstream schema generation fails."""
    if root is not None:
        return create_model(model_name, root=(Any, None))

    fields = {
        name: (Any, None) for name in (field_definitions or {})
    } or {"output": (Any, None)}
    return create_model(model_name, **fields)


def apply_compat_patches() -> None:
    """Patch upstream schema generation bugs used by deepagents/langgraph.

    This keeps normal behavior intact and only falls back to permissive `Any`
    fields when langgraph cannot materialize a model because of unsupported
    schema annotations such as `OmitFromSchema` + `NotRequired`.
    """
    try:
        from langgraph._internal import _pydantic as langgraph_pydantic
    except ImportError:
        return

    if getattr(langgraph_pydantic.create_model, "_langchain_oci_patched", False):
        return

    original_create_model = langgraph_pydantic.create_model

    def patched_create_model(
        model_name: str,
        *,
        field_definitions: dict[str, Any] | None = None,
        root: Any | None = None,
    ) -> type[BaseModel]:
        try:
            return original_create_model(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )
        except Exception as ex:
            if ex.__class__.__name__ != "PydanticForbiddenQualifier":
                raise
            return _build_loose_schema(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )

    patched_create_model._langchain_oci_patched = True  # type: ignore[attr-defined]
    langgraph_pydantic.create_model = patched_create_model
