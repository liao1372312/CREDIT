from credit.toolsandbox_m_contract_fast_v3 import (
    _role_word,
    explicit_add_contact_arguments,
    fast_contract_violation,
    propose_fast_repair,
)


def test_extracts_disclosed_contact_wordings():
    requests = (
        ("Create an auditor contact called \"Recreate Latest\" at +14155550202.",
         {"name": "Recreate Latest", "phone_number": "+14155550202",
          "relationship": "auditor"}),
        ("Record +14155550202 as auditor under \"Recreate Latest\".",
         {"name": "Recreate Latest", "phone_number": "+14155550202",
          "relationship": "auditor"}),
        ("Save a coordinator contact named 'Substitute Reminder' with number "
         "+14155550203.",
         {"name": "Substitute Reminder", "phone_number": "+14155550203",
          "relationship": "coordinator"}),
        ("Add a contact named 'Substitute Reminder' at +14155550203 as "
         "coordinator.",
         {"name": "Substitute Reminder", "phone_number": "+14155550203",
          "relationship": "coordinator"}),
        ("Save Ops Desk (+14155550182) as a coordinator contact.",
         {"name": "Ops Desk", "phone_number": "+14155550182",
          "relationship": "coordinator"}),
    )
    for query, expected in requests:
        assert explicit_add_contact_arguments(query) == expected


def test_repairs_omitted_and_swapped_contact_fields():
    query = (
        "Create an auditor contact called \"Recreate Latest\" at +14155550202, "
        "send exactly \"rebuild next reminder\", and disable cellular."
    )
    allowed = {"add_contact", "send_message_with_phone_number",
               "set_cellular_service_status"}
    for arguments in (
        {"name": "Recreate Latest", "phone_number": "+14155550202"},
        {"name": "auditor", "phone_number": "+14155550202",
         "relationship": "Recreate Latest"},
    ):
        action = {"tool": "add_contact", "arguments": arguments}
        violation = fast_contract_violation(action, [], query, allowed)
        assert "explicit public-request fields" in violation
        repaired, reason = propose_fast_repair(action, [], query, allowed)
        assert repaired == {"tool": "add_contact", "arguments": {
            "name": "Recreate Latest", "phone_number": "+14155550202",
            "relationship": "auditor"}}
        assert "restore explicit contact fields" in reason


def test_valid_explicit_contact_is_unchanged():
    query = "Save Ops Desk (+14155550182) as a coordinator contact."
    allowed = {"add_contact", "search_contacts", "modify_contact"}
    action = {"tool": "add_contact", "arguments": {
        "name": "Ops Desk", "phone_number": "+14155550182",
        "relationship": "coordinator"}}
    assert fast_contract_violation(action, [], query, allowed) is None
    assert propose_fast_repair(action, [], query, allowed) is None


def test_mistaken_role_creation_first_resolves_existing_contact():
    query = (
        "Mark my boss as manager and save Ops Desk (+14155550182) as a "
        "coordinator contact."
    )
    allowed = {"search_contacts", "modify_contact", "add_contact"}
    action = {"tool": "add_contact", "arguments": {
        "name": "boss", "phone_number": "+11233344455",
        "relationship": "manager"}}
    repaired, reason = propose_fast_repair(action, [], query, allowed)
    assert repaired == {"tool": "search_contacts",
                        "arguments": {"relationship": "boss"}}
    assert "existing role contact" in reason


def test_role_resolution_ignores_latest_message_phrase():
    query = (
        "Use my most recent text unchanged, mark my boss as manager, and save "
        "Ops Desk (+14155550182) as a coordinator contact."
    )
    assert _role_word(query) == "boss"
