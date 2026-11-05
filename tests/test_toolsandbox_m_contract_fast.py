from types import SimpleNamespace

from credit.toolsandbox_m_contract_fast import (
    fast_contract_violation,
    fast_select,
    propose_fast_repair,
    request_required_mutations,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def scenario(tools):
    return SimpleNamespace(starting_context=SimpleNamespace(tool_allow_list=tools))


def call(tool, response, **arguments):
    return {"tool": tool, "arguments": arguments, "response": repr(response),
            "exception": None}


def test_repair_inserts_current_time_before_latest_message_search():
    repair = propose_fast_repair(
        {"tool": "search_messages", "arguments": {}}, [],
        "Use my newest message as a reminder.",
        {"get_current_timestamp", "search_messages", "add_reminder"},
    )
    assert repair[0] == {"tool": "get_current_timestamp", "arguments": {}}


def test_repair_latest_message_search_uses_upper_bound():
    repair = propose_fast_repair(
        {"tool": "search_messages", "arguments": {"creation_timestamp_lowerbound": 100}},
        [call("get_current_timestamp", 100)],
        "Use my latest message as a reminder.",
        {"get_current_timestamp", "search_messages", "add_reminder"},
    )
    assert repair[0]["arguments"] == {"creation_timestamp_upperbound": 100}


def test_repair_inserts_shift_for_early_datetime_conversion():
    repair = propose_fast_repair(
        {"tool": "datetime_info_to_timestamp", "arguments": {
            "year": 2026, "month": 6, "day": 18, "hour": 9}},
        [call("get_current_timestamp", 100)], "Remind me tomorrow at 9.",
        {"get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "add_reminder"},
    )
    assert repair[0] == {"tool": "shift_timestamp",
                         "arguments": {"timestamp": 100, "days": 1}}


def test_repair_calendar_date_preserves_proposed_clock_time():
    calls = [call("get_current_timestamp", 100),
             call("shift_timestamp", 200, timestamp=100, days=1),
             call("timestamp_to_datetime_info", {
                 "year": 2026, "month": 10, "day": 19,
                 "hour": 12, "minute": 0, "second": 0}, timestamp=200)]
    repair = propose_fast_repair(
        {"tool": "datetime_info_to_timestamp", "arguments": {
            "year": 2026, "month": 6, "day": 18,
            "hour": 9, "minute": 15, "second": 0}},
        calls, "Remind me tomorrow at 9:15.",
        {"get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "add_reminder"},
    )
    assert repair[0]["arguments"] == {
        "year": 2026, "month": 10, "day": 19,
        "hour": 9, "minute": 15, "second": 0}


def test_repair_reminder_uses_observed_latest_content_and_timestamp():
    calls = [call("get_current_timestamp", 100),
             call("shift_timestamp", 200),
             call("timestamp_to_datetime_info", {
                 "year": 2026, "month": 10, "day": 19}),
             call("datetime_info_to_timestamp", 300),
             call("search_messages", [{"content": "raw", "creation_timestamp": 90,
                                        "sender_phone_number": "+1"}])]
    repair = propose_fast_repair(
        {"tool": "add_reminder", "arguments": {
            "content": "Reminder: raw", "reminder_timestamp": 999}},
        calls, "Use my latest message as a reminder tomorrow.",
        {"get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
         "datetime_info_to_timestamp", "search_messages", "add_reminder"},
    )
    assert repair[0]["arguments"] == {"content": "raw", "reminder_timestamp": 300}


def test_repair_role_lookup_uses_relationship_field():
    repair = propose_fast_repair(
        {"tool": "search_contacts", "arguments": {"name": "boss"}}, [],
        "Mark my boss relationship as manager.",
        {"search_contacts", "modify_contact"},
    )
    assert repair[0] == {"tool": "search_contacts",
                         "arguments": {"relationship": "boss"}}


def test_repair_role_lookup_handles_mark_as_wording():
    repair = propose_fast_repair(
        {"tool": "search_contacts", "arguments": {"name": "boss"}}, [],
        "Mark my boss as manager too.",
        {"search_contacts", "modify_contact"},
    )
    assert repair[0] == {"tool": "search_contacts",
                         "arguments": {"relationship": "boss"}}


def test_repair_final_answer_completes_uniquely_resolved_contact_update():
    calls = [call("search_contacts", [{
        "person_id": "p1", "name": "Homer", "relationship": "boss"}])]
    repair = propose_fast_repair(
        {"answer": "done"}, calls, "Change my boss relationship to manager.",
        {"search_contacts", "modify_contact"},
    )
    assert repair[0] == {"tool": "modify_contact",
                         "arguments": {"person_id": "p1", "relationship": "manager"}}


def test_ambiguous_contact_is_not_auto_repaired():
    calls = [call("search_contacts", [
        {"person_id": "p1", "name": "A", "relationship": "boss"},
        {"person_id": "p2", "name": "B", "relationship": "boss"},
    ])]
    repair = propose_fast_repair(
        {"answer": "done"}, calls, "Change my boss relationship to manager.",
        {"search_contacts", "modify_contact"},
    )
    assert repair is None


def test_adaptive_selector_omits_contract_on_neutral_request():
    chosen = fast_select([], "Turn off low battery mode.",
                         scenario(["set_low_battery_mode_status"]), TinyTokenizer())
    assert chosen == []


def test_adaptive_selector_keeps_contract_for_relative_date():
    chosen = fast_select([], "Remind me tomorrow at 9.", scenario([
        "get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
        "datetime_info_to_timestamp", "add_reminder"]), TinyTokenizer())
    assert chosen[0]["memory_schema"] == "request_derived_execution_contract_v1"


def test_wide_allowlist_does_not_create_decoy_mutation_requirements():
    allowed = {"search_contacts", "send_message_with_phone_number",
               "modify_contact", "add_reminder", "set_wifi_status"}
    assert request_required_mutations("Send 'hi' to Ada.", allowed) == {
        "send_message_with_phone_number"}
    violation = fast_contract_violation(
        {"answer": "done"}, [], "Send 'hi' to Ada.", allowed)
    assert violation.endswith("send_message_with_phone_number")
    assert "modify_contact" not in violation
    assert fast_contract_violation(
        {"answer": "done"}, [call("send_message_with_phone_number", None)],
        "Send 'hi' to Ada.", allowed) is None


def test_quoted_replace_payload_does_not_activate_replacement():
    allowed = {"search_contacts", "send_message_with_phone_number",
               "remove_contact", "add_contact"}
    assert request_required_mutations(
        "Send the exact text 'replace draft' to Ada.", allowed) == {
            "send_message_with_phone_number"}


def test_mark_role_request_requires_contact_modification():
    allowed = {"search_contacts", "modify_contact", "add_reminder"}
    assert request_required_mutations("Mark my boss as manager.", allowed) == {
        "modify_contact"}


def test_mixed_contact_replacement_and_new_reminder_has_no_decoy_removal():
    allowed = {"search_contacts", "remove_contact", "add_contact",
               "search_reminder", "remove_reminder", "add_reminder"}
    required = request_required_mutations(
        "Replace John Petrucci with John P at the same number, and remind me "
        "tomorrow to call John P.", allowed)
    assert required == {"remove_contact", "add_contact", "add_reminder"}


def test_reminder_replacement_does_not_require_contact_replacement():
    allowed = {"search_contacts", "remove_contact", "add_contact",
               "search_reminder", "remove_reminder", "add_reminder"}
    required = request_required_mutations(
        "Replace my next reminder with 'Finalize slides' at the same time.", allowed)
    assert required == {"remove_reminder", "add_reminder"}


def test_make_message_a_reminder_requires_add():
    allowed = {"search_messages", "add_reminder", "remove_reminder"}
    assert request_required_mutations(
        "Make my newest message a reminder for tomorrow.", allowed) == {
            "add_reminder"}


def test_possessive_apostrophes_do_not_hide_later_send_command():
    allowed = {"add_contact", "modify_contact", "send_message_with_phone_number"}
    required = request_required_mutations(
        "Set Fredrik's relationship to bandmate, then send Let's talk to the lead.",
        allowed)
    assert required == {"modify_contact", "send_message_with_phone_number"}


def test_set_role_to_wording_requires_contact_modification():
    assert request_required_mutations(
        "Set my boss to manager.", {"search_contacts", "modify_contact"}) == {
            "modify_contact"}


def test_global_latest_search_removes_unrequested_recipient_filter():
    calls = [call("get_current_timestamp", 100)]
    action = {"tool": "search_messages", "arguments": {
        "recipient_phone_number": "+1", "creation_timestamp_upperbound": 100}}
    query = "Use my newest message as a reminder tomorrow."
    allowed = {"get_current_timestamp", "search_messages", "add_reminder"}
    assert "must not add" in fast_contract_violation(action, calls, query, allowed)
    repair = propose_fast_repair(action, calls, query, allowed)
    assert repair[0] == {"tool": "search_messages", "arguments": {
        "creation_timestamp_upperbound": 100}}


def test_source_sender_is_resolved_before_contact_creation():
    query = "Save the GPU-offer sender as Hardware Lead."
    allowed = {"search_messages", "add_contact"}
    action = {"tool": "add_contact", "arguments": {
        "name": "Hardware Lead", "phone_number": "+wrong"}}
    repair = propose_fast_repair(action, [], query, allowed)
    assert repair[0] == {"tool": "search_messages", "arguments": {"content": "gpu"}}

    calls = [call("search_messages", [{
        "content": "GPU-offer", "sender_phone_number": "+right",
        "creation_timestamp": 90}], content="gpu")]
    repair = propose_fast_repair(action, calls, query, allowed)
    assert repair[0]["arguments"]["phone_number"] == "+right"
