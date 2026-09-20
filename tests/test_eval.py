from __future__ import annotations

from phase1.eval_harness import run_eval, seed_eval_fixture
from phase1.embed import set_embedder
from tests.fakes import HashEmbedder


REQUIRED_SUMMARY = (
    "precision_at_k",
    "recall_at_k",
    "top1_correctness",
    "top3_correctness",
    "irrelevant_memory_rate",
    "wrong_project_rate",
    "stale_superseded_memory_rate",
    "contradiction_rate",
    "approx_token_cost",
    "zero_result_correctness",
)

ADVERSARIAL_IDS = {
    "adv-paraphrase-same-fact",
    "adv-similar-different-decisions",
    "adv-correction-over-old",
    "adv-unrelated-high-importance",
    "adv-vague-query",
    "adv-zero-result",
}


def test_eval_metrics_and_adversarial_cases(db_path):
    set_embedder(HashEmbedder())
    seed_eval_fixture(db_path)
    payload = run_eval(db_path=db_path)
    summary = payload["summary"]
    for key in REQUIRED_SUMMARY:
        assert key in summary, key
    ids = {r["id"] for r in payload["results"]}
    assert ADVERSARIAL_IDS <= ids
    zero = next(r for r in payload["results"] if r["id"] == "adv-zero-result")
    assert zero["expect_empty"] is True
    assert zero["zero_result_correct"] is True
    stale = next(r for r in payload["results"] if r["id"] == "q-stale-must-stay-out")
    assert stale["stale_superseded_rate"] == 0.0
    # Unrelated high-importance snack must not outrank the store decision.
    unr = next(r for r in payload["results"] if r["id"] == "adv-unrelated-high-importance")
    assert "mem-unrelated-high" not in unr["injected_ids"][:1]
