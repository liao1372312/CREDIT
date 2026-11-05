from types import SimpleNamespace

from credit.toolsandbox_m_contract_selector import (
    contract_select,
    contract_violation,
    derive_execution_contract,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def scenario(tools):
    return SimpleNamespace(starting_context=SimpleNamespace(tool_allow_list=tools))


def test_recreate_contract_requires_contact_removal_before_addition():
    rules = " ".join(derive_execution_contract(
        "Recreate Ada as Ada L and keep her phone.",
        {"search_contacts", "remove_contact", "add_contact"},
    ))
    assert "remove_contact" in rules
    assert "adding a second record alone is not a replacement" in rules
    assert rules.index("search_contacts") < rules.index("remove_contact") < rules.index("add_contact")


def test_tomorrow_contract_anchors_once_to_current_time():
    rules = " ".join(derive_execution_contract(
        "Remind me tomorrow at 9 AM.",
        {"get_current_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "add_reminder"},
    ))
    assert "exact returned timestamp unchanged" in rules
    assert "exactly 1 day(s)" in rules
    assert "Do not pre-shift" in rules


def test_tomorrow_contract_prefers_one_shift_when_tool_is_available():
    rules = " ".join(derive_execution_contract(
        "Remind me tomorrow at 9 AM.",
        {"get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "add_reminder"},
    ))
    assert "one shift_timestamp call with days=1" in rules
    assert "keep its returned year/month/day unchanged" in rules
    assert "move the returned calendar date forward" not in rules


def test_latest_message_contract_uses_current_time_as_upper_bound():
    rules = " ".join(derive_execution_contract(
        "Use my newest message as a reminder and send it prefixed 'Note: '.",
        {"get_current_timestamp", "search_messages", "add_reminder",
         "send_message_with_phone_number"},
    ))
    assert "creation_timestamp_upperbound" in rules
    assert "greatest creation_timestamp" in rules
    assert "apply any prefix" in rules


def test_next_reminder_contract_uses_first_future_result():
    rules = " ".join(derive_execution_contract(
        "Recreate my next reminder with new text.",
        {"get_current_timestamp", "search_reminder", "remove_reminder",
         "add_reminder"},
    ))
    assert "reminder_timestamp_lowerbound" in rules
    assert "smallest reminder_timestamp" in rules
    assert "remove_reminder" in rules


def test_contract_selection_is_bounded_and_keeps_one_recipe():
    pool = [{
        "uid": "recipe", "text": "Verified recipe.", "tools": ["add_contact"],
        "tokens": 2, "source_task": "old", "source_upstream_score": 1.0,
        "source_requested_state_verified": True,
    }]
    chosen = contract_select(
        pool,
        "Recreate Ada and keep her phone.",
        scenario(["search_contacts", "remove_contact", "add_contact"]),
        TinyTokenizer(),
    )
    assert len(chosen) == 2
    assert chosen[0]["memory_schema"] == "request_derived_execution_contract_v1"
    assert chosen[1]["uid"] == "recipe"
    assert sum(item["tokens"] for item in chosen) <= 512


def test_shift_contract_rejects_incompatible_manual_date_recipe():
    pool = [{
        "uid": "manual-date", "text": "Manual date recipe.",
        "tools": ["get_current_timestamp", "timestamp_to_datetime_info",
                  "datetime_info_to_timestamp", "add_reminder"],
        "tokens": 3, "source_task": "old", "source_upstream_score": 1.0,
        "source_requested_state_verified": True,
    }]
    chosen = contract_select(
        pool,
        "Remind me tomorrow at 9 AM.",
        scenario(["get_current_timestamp", "shift_timestamp",
                  "timestamp_to_datetime_info", "datetime_info_to_timestamp",
                  "add_reminder"]),
        TinyTokenizer(),
    )
    assert len(chosen) == 1
    assert chosen[0]["memory_schema"] == "request_derived_execution_contract_v1"


def test_runtime_contract_rejects_wrong_relative_date_sequence():
    calls = [{"tool": "get_current_timestamp", "response": "100.0", "exception": None}]
    violation = contract_violation(
        {"tool": "timestamp_to_datetime_info", "arguments": {"timestamp": 100.0}},
        calls, "Remind me tomorrow at 9 AM.",
        {"get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "add_reminder"},
    )
    assert "shifted timestamp" in violation


def test_runtime_contract_rejects_reply_to_recipient():
    calls = [
        {"tool": "get_current_timestamp", "response": "100.0", "exception": None},
        {"tool": "search_messages", "response": repr([{
            "content": "hello", "creation_timestamp": 99.0,
            "sender_phone_number": "+1", "recipient_phone_number": "+2",
        }]), "exception": None},
    ]
    violation = contract_violation(
        {"tool": "send_message_with_phone_number",
         "arguments": {"phone_number": "+2", "content": "ok"}},
        calls, "Reply to whoever sent my latest message.",
        {"get_current_timestamp", "search_messages", "send_message_with_phone_number"},
    )
    assert "sender_phone_number" in violation
