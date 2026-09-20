from __future__ import annotations

from phase1.privacy import (
    DESTINATION_EXTERNAL,
    DESTINATION_LOCAL,
    allowed_for_auto_retrieve,
    allowed_for_destination,
    allowed_for_external_model,
    normalize_flags,
)


def test_privacy_defaults_allow_local_retrieve():
    flags = normalize_flags({})
    assert flags["remote_ok"] == 1
    assert allowed_for_auto_retrieve(flags)
    assert allowed_for_destination(flags, DESTINATION_LOCAL)


def test_local_only_and_sensitive_blocked_from_external():
    row = normalize_flags(
        {
            "remote_ok": 0,
            "local_only": 1,
            "sensitive": 1,
            "never_send_external_model": 1,
        }
    )
    assert allowed_for_destination(row, DESTINATION_LOCAL)
    assert not allowed_for_external_model(row)
    assert not allowed_for_destination(row, DESTINATION_EXTERNAL)


def test_never_auto_retrieve():
    row = normalize_flags({"never_auto_retrieve": 1})
    assert not allowed_for_auto_retrieve(row)
    assert not allowed_for_destination(row, DESTINATION_LOCAL, auto_retrieve=True)
    assert allowed_for_destination(row, DESTINATION_LOCAL, auto_retrieve=False)
