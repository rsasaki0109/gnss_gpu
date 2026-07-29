"""Repository contract for local experiment artifacts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wp29_to_wp31_are_explicitly_local_workspaces():
    ignore_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert {"results/wp29/", "results/wp30/", "results/wp31/"} <= ignore_lines


def test_result_policy_classifies_each_workspace():
    policy = (ROOT / "results" / "ARTIFACT_POLICY.md").read_text(encoding="utf-8")
    for workspace in ("wp29", "wp30", "wp31"):
        assert f"`{workspace}`" in policy
    assert "What is committed" in policy
    assert "What remains local" in policy
