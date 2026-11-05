from credit.toolsandbox_m_selector_attribution import METHODS, PRIMARY_CONTRASTS, select


class Scenario:
    class Context:
        tool_allow_list = ["search_contacts", "add_contact", "set_wifi_status"]

    starting_context = Context()


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


POOL = [
    {"uid": "old", "tools": ["search_contacts"], "tokens": 3,
     "arrival": "a", "source_task": "find contact",
     "source_upstream_score": 1.0, "text": "find contact",
     "memory_schema": "verified_recipe_v2"},
    {"uid": "new", "tools": ["add_contact"], "tokens": 3,
     "arrival": "b", "source_task": "add contact",
     "source_upstream_score": 1.0, "text": "add contact",
     "memory_schema": "verified_recipe_v2"},
]


def test_registered_design_has_fixed_executor_comparisons():
    assert len(METHODS) == 8
    assert ("CREDIT/v3", "NoMem/v3",
            "historical memory under fixed v3 executor") in PRIMARY_CONTRASTS
    assert ("NoMem/v3", "NoMem/Base",
            "argument-aware executor with empty memory") in PRIMARY_CONTRASTS


def test_empty_and_fifo_selectors_are_distinct():
    scenario = Scenario()
    tokenizer = Tokenizer()
    assert select("NoMem/v3", POOL, "task", "add contact", scenario,
                  set(), tokenizer) == []
    fifo = select("FIFO/v3", POOL, "task", "add contact", scenario,
                  set(), tokenizer)
    assert [row["uid"] for row in fifo] == ["old", "new"]
