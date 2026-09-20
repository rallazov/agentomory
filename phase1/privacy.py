"""Local visibility controls: what may be auto-retrieved or sent to an external model."""
from __future__ import annotations

from typing import Any, Mapping, Optional

DESTINATION_LOCAL = "local"
DESTINATION_EXTERNAL = "external_model"

# Flags the local service understands. Defaults keep existing memories retrievable.
DEFAULT_FLAGS = {
    "remote_ok": 1,
    "local_only": 0,
    "sensitive": 0,
    "never_auto_retrieve": 0,
    "never_send_external_model": 0,
}


def _flag(row: Mapping[str, Any], name: str, default: int = 0) -> int:
    if name not in row.keys() if hasattr(row, "keys") else name not in row:
        return default
    val = row[name]
    if val is None:
        return default
    return int(val)


def normalize_flags(candidate: Mapping[str, Any] | None) -> dict:
    src = candidate or {}
    return {
        "remote_ok": int(src.get("remote_ok", DEFAULT_FLAGS["remote_ok"])),
        "local_only": int(src.get("local_only", DEFAULT_FLAGS["local_only"])),
        "sensitive": int(src.get("sensitive", DEFAULT_FLAGS["sensitive"])),
        "never_auto_retrieve": int(
            src.get("never_auto_retrieve", DEFAULT_FLAGS["never_auto_retrieve"])
        ),
        "never_send_external_model": int(
            src.get("never_send_external_model", DEFAULT_FLAGS["never_send_external_model"])
        ),
    }


def allowed_for_auto_retrieve(row: Mapping[str, Any]) -> bool:
    return _flag(row, "never_auto_retrieve", 0) == 0


def allowed_for_external_model(row: Mapping[str, Any]) -> bool:
    if _flag(row, "never_send_external_model", 0):
        return False
    if _flag(row, "local_only", 0):
        return False
    if _flag(row, "sensitive", 0):
        return False
    if _flag(row, "remote_ok", 1) == 0:
        return False
    return True


def allowed_for_destination(
    row: Mapping[str, Any],
    destination: Optional[str] = DESTINATION_LOCAL,
    *,
    auto_retrieve: bool = True,
) -> bool:
    dest = destination or DESTINATION_LOCAL
    if auto_retrieve and not allowed_for_auto_retrieve(row):
        return False
    if dest == DESTINATION_EXTERNAL:
        return allowed_for_external_model(row)
    return True


def sql_privacy_clause(destination: Optional[str] = DESTINATION_LOCAL, alias: str = "m") -> str:
    """SQL fragment (AND …) restricting memories for a destination. Columns may be missing on very old DBs — call after migrate."""
    parts = [f"COALESCE({alias}.never_auto_retrieve, 0) = 0"]
    if (destination or DESTINATION_LOCAL) == DESTINATION_EXTERNAL:
        parts.extend(
            [
                f"COALESCE({alias}.remote_ok, 1) = 1",
                f"COALESCE({alias}.local_only, 0) = 0",
                f"COALESCE({alias}.sensitive, 0) = 0",
                f"COALESCE({alias}.never_send_external_model, 0) = 0",
            ]
        )
    return " AND " + " AND ".join(parts)
