from types import SimpleNamespace

from credit.toolsandbox_m_scoped_compose_selector import (
    scoped_compose_select,
    scoped_contract_select,
    scoped_full_select,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


SCENARIO = SimpleNamespace(starting_context=SimpleNamespace(tool_allow_list=[
    "search_contacts", "add_contact", "send_message_with_phone_number",
    "search_reminder", "add_reminder", "set_wifi_status",
    "set_cellular_service_status", "remove_contact",
]))

POOL = [
    {"uid": "contact-message", "tools": ["add_contact", "send_message_with_phone_number"],
     "tokens": 12, "text": "add and send", "arrival": "a",
     "source_task": "a", "source_upstream_score": 1.0},
    {"uid": "reminder", "tools": ["add_reminder"], "tokens": 8,
     "text": "add reminder", "arrival": "b", "source_task": "b",
     "source_upstream_score": 1.0},
    {"uid": "irrelevant", "tools": ["remove_contact"], "tokens": 5,
     "text": "remove contact", "arrival": "c", "source_task": "c",
     "source_upstream_score": 1.0},
]


def test_scoping_uses_public_request_in_wide_catalogue():
    query = (
        "Create an auditor contact named 'Archive' at +14155550901, text it "
        "exactly 'ready', and create a reminder called 'review'."
    )
    selected = scoped_compose_select(POOL, query, SCENARIO)
    assert [row["uid"] for row in selected] == ["contact-message", "reminder"]
    assert "irrelevant" not in {row["uid"] for row in selected}


def test_full_selection_adds_contract_without_hidden_state():
    query = "Disable Wi-Fi and mobile data."
    tokenizer = Tokenizer()
    contract = scoped_contract_select(query, SCENARIO, tokenizer)
    full = scoped_full_select(POOL, query, SCENARIO, tokenizer)
    assert len(contract) == 1
    assert contract[0]["memory_schema"] == "request_derived_execution_contract_v1"
    assert full[0] == contract[0]
