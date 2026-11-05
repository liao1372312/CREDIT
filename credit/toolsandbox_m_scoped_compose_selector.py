"""Public-request-scoped composition selection for ToolSandbox-M.

The original composition selector scores every tool in the exposed catalogue.
That is uninformative when all methods receive the same wide catalogue.  This
module first derives the operations named by the public request, then applies
the unchanged schema-coverage rule to that request-scoped view.  It never reads
scenario milestones, evaluator state, or hidden target arguments.
"""

from __future__ import annotations

from types import SimpleNamespace

from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_fast import request_scoped_tools
from .toolsandbox_m_contract_selector import contract_note


def _proxy(query: str, scenario):
    allowed = set(scenario.starting_context.tool_allow_list or [])
    scoped = sorted(request_scoped_tools(query, allowed))
    return SimpleNamespace(
        starting_context=SimpleNamespace(tool_allow_list=scoped)
    )


def scoped_compose_select(pool: list[dict], query: str, scenario) -> list[dict]:
    """Select at most two historical recipes for public-request operations."""
    return compose_select(pool, query, _proxy(query, scenario))


def scoped_contract_select(query: str, scenario, tokenizer) -> list[dict]:
    """Create a contract over only the operations inferred from the request."""
    return [contract_note(query, _proxy(query, scenario), tokenizer)]


def scoped_full_select(pool: list[dict], query: str, scenario,
                       tokenizer, capacity: int = 512) -> list[dict]:
    """Pair the scoped contract with every fitting scoped recipe (at most two)."""
    selected = scoped_contract_select(query, scenario, tokenizer)
    for note in scoped_compose_select(pool, query, scenario):
        if sum(item["tokens"] for item in [*selected, note]) <= capacity:
            selected.append(note)
    return selected
