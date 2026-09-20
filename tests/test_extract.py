from __future__ import annotations

from phase1.extract import (
    canonicalize,
    extract_candidates,
    extract_temporal,
    looks_memory_worthy,
)


MULTI = (
    "We decided to use SQLite for the local store. "
    "Never send calendar tokens to any cloud model. "
    "By November we need twenty test users."
)


def test_prefilter_skips_greetings():
    assert not looks_memory_worthy("ok")
    assert not looks_memory_worthy("thanks")
    assert looks_memory_worthy(MULTI)


def test_one_message_yields_multiple_atomic_memories():
    cands = extract_candidates(MULTI, role="user", message_id="m1", conversation_id="c1")
    assert len(cands) >= 2
    types = {c["memory_type"] for c in cands}
    texts = " ".join(c["canonical_text"].lower() for c in cands)
    assert "sqlite" in texts
    assert "token" in texts or "cloud" in texts
    assert any(t in types for t in ("decision", "fact", "constraint", "goal"))
    for c in cands:
        assert c["canonical_text"] != MULTI
        assert c["canonical_text"] != MULTI[:800]
        assert len(c["canonical_text"]) < len(MULTI)
        assert c["source_message_id"] == "m1"
        assert c["origin"] == "user_stated"
        assert "explicit" in c
        assert "appears_to_correct" in c
        assert c["memory_type"]
        assert 0 <= c["importance"] <= 1
        assert 0 <= c["confidence"] <= 1


def test_canonical_is_not_truncated_original():
    long_msg = (
        "Hey so I just wanted to mention, also, by the way, "
        "we decided to use SQLite for the local store, "
        "and I guess that's the main thing I wanted to lock in today "
        "after rambling about the weather and lunch plans for a while."
    )
    cands = extract_candidates(long_msg, role="user", message_id="m2")
    assert cands
    for c in cands:
        assert c["canonical_text"] != long_msg[:800]
        assert "rambling about the weather" not in c["canonical_text"].lower()
        assert "sqlite" in c["canonical_text"].lower()


def test_correction_flag_and_temporal():
    text = "That's wrong — we actually use SQLite for local memory by November."
    cands = extract_candidates(text, role="user", message_id="m3")
    assert cands
    assert any(c["appears_to_correct"] for c in cands)
    temporal = extract_temporal(text)
    assert temporal and "deadline" in temporal
    assert any(c.get("temporal_meaning") for c in cands)


def test_canonicalize_rewrites_decision():
    out = canonicalize("We decided to use SQLite for the local store", "decision")
    assert out.lower().startswith("use sqlite")
    assert not out.lower().startswith("we decided")


def test_assistant_inferred_is_labeled():
    text = "Decision: we agreed the standing rule is story-bound launches only."
    cands = extract_candidates(text, role="assistant", message_id="a1")
    assert cands
    assert all(c["origin"] == "agent_inferred" for c in cands)
    assert all(c["explicit"] is False for c in cands)
    assert all(c["confidence"] <= 0.45 for c in cands)
