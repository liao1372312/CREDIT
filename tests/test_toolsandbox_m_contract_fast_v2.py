from credit.toolsandbox_m_contract_fast_v2 import (
    fast_contract_violation,
    propose_fast_repair,
)


def test_quoted_replace_literals_do_not_block_contact_creation():
    query = (
        "Add a contact named 'Latest Replace' at +14155550183 as reviewer, "
        "send that contact the exact text 'replace latest draft', and turn "
        "Wi-Fi off."
    )
    allowed = {
        "search_contacts", "add_contact", "remove_contact",
        "send_message_with_phone_number", "set_wifi_status",
    }
    action = {"tool": "add_contact", "arguments": {
        "name": "Latest Replace", "phone_number": "+14155550183",
        "relationship": "reviewer",
    }}
    assert fast_contract_violation(action, [], query, allowed) is None
    assert propose_fast_repair(action, [], query, allowed) is None


def test_unquoted_replacement_still_requires_removal():
    query = "Replace John Petrucci with JP Archive at the same number."
    allowed = {"search_contacts", "add_contact", "remove_contact"}
    action = {"tool": "add_contact", "arguments": {
        "name": "JP Archive", "phone_number": "+1234560987",
    }}
    violation = fast_contract_violation(action, [], query, allowed)
    assert violation == (
        "contact replacement requires a successful remove_contact before "
        "add_contact"
    )


def test_final_answer_still_requires_requested_mutations():
    query = "Send the exact text 'replace draft' to Ada and disable wifi."
    allowed = {"send_message_with_phone_number", "set_wifi_status"}
    violation = fast_contract_violation({"answer": "done"}, [], query, allowed)
    assert "send_message_with_phone_number" in violation
    assert "set_wifi_status" in violation
