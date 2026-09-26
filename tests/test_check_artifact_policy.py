"""Pure artifact-policy tests: no repository index or subprocess required."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_artifact_policy", Path(__file__).resolve().parents[1] / "scripts/ci/check_artifact_policy.py"
)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)

# Construct real-world boundary examples without citing an actual workspace.
SHORT = "wp" + "2"
LONG = SHORT + "8"


def classify(artifacts, files=(), allowlist=""):
    return policy.classify_artifacts(
        artifacts, policy.build_corpus(files), policy.parse_allowlist(allowlist)
    )


@pytest.mark.parametrize("root", policy.RESULT_ROOTS)
def test_exact_path_substring(root):
    artifact = root + "root_fixture.csv"
    assert classify([artifact], [("guide.md", f"prefix{artifact}suffix".encode())]) == ([artifact], [])


@pytest.mark.parametrize("ending", ["/child", '"', "'", " ", "\t", "\nnext", ")", "\x60", ""])
@pytest.mark.parametrize("root", policy.RESULT_ROOTS)
def test_directory_boundaries(root, ending):
    artifact = root + SHORT + "/sub/file.csv"
    assert classify([artifact], [("guide.md", (root + SHORT + ending).encode())]) == ([artifact], [])


@pytest.mark.parametrize("root", policy.RESULT_ROOTS)
def test_workspace_prefix_does_not_match_longer_name(root):
    artifact = root + SHORT + "/file.csv"
    for ending in ("8", "_extra", "-extra", ".csv", ":"):
        assert classify([artifact], [("guide.md", (root + SHORT + ending).encode())]) == ([], [artifact])
    longer = root + LONG + "/file.csv"
    assert classify([artifact, longer], [("guide.md", (root + LONG).encode())]) == ([longer], [artifact])


def test_ancestors_exclude_result_root():
    path = "experiments/results/fixture_alpha/sub/file.csv"
    assert policy.ancestors(path) == (
        "experiments/results/fixture_alpha", "experiments/results/fixture_alpha/sub"
    )
    assert classify([path], [("guide.md", b"experiments/results/ results/")]) == ([], [path])
    assert policy.ancestors("results/file.csv") == ()
    assert policy.ancestors("docs/file.csv") == ()
    assert policy.workspace("results/file.csv") == "results/(root)"


@pytest.mark.parametrize("suffix", [".py", ".sh", ".yml"])
def test_constructed_code_reference(suffix):
    artifact = "results/fixture_alpha/sub/file.csv"
    assert classify([artifact], [("script" + suffix, b'Path("results") / "fixture_alpha"')]) == ([artifact], [])


@pytest.mark.parametrize("text", [b"fixture_alphabet results", b"xfixture_alpha results", b"fixture_alpha"])
def test_code_requires_whole_word_and_results(text):
    artifact = "results/fixture_alpha/file.csv"
    assert classify([artifact], [("script.py", text)]) == ([], [artifact])


def test_code_requires_same_file_and_supported_suffix():
    artifact = "results/fixture_alpha/file.csv"
    files = [("one.py", b"results"), ("two.py", b"fixture_alpha"), ("guide.md", b"results fixture_alpha")]
    assert classify([artifact], files) == ([], [artifact])
    # The numeric boundary also applies to code references.
    artifact = "results/" + SHORT + "/file.csv"
    assert classify([artifact], [("script.py", ("results " + LONG).encode())]) == ([], [artifact])


def test_allowlist_globs_and_comments():
    allowlist = "# justified exception\n results/fixture_alpha/*.csv # dependency\n\nexperiments/results/item?.json\n"
    assert policy.parse_allowlist(allowlist) == (
        "results/fixture_alpha/*.csv", "experiments/results/item?.json"
    )
    kept = ["experiments/results/item1.json", "results/fixture_alpha/sub/a.csv"]
    missing = ["experiments/results/item12.json", "results/fixture_alpha/a.CSV", "results/fixture_beta/a.csv"]
    assert classify(kept + missing, allowlist=allowlist) == (kept, missing)


def test_corpus_excludes_all_result_files():
    artifact = "results/fixture_alpha/file.csv"
    files = [(path, artifact.encode()) for path in (
        "results/report.md", "experiments/results/report.py",
        policy.ALLOWLIST, "results/ARTIFACT_POLICY.md"
    )]
    assert policy.build_corpus(files) == {}
    assert classify([artifact], files) == ([], [artifact])
    assert policy.build_corpus([("docs/results/guide.md", b"text")]) == {"docs/results/guide.md": "text"}


def test_corpus_filters_binary_oversize_and_undecodable():
    files = [
        ("nul.bin", b"text\0data"), ("control.bin", b"\x01abc"),
        ("bad.txt", b"\xff"), ("large.txt", b"a" * (policy.MAX_TEXT_BYTES + 1)),
        ("limit.txt", b"a" * policy.MAX_TEXT_BYTES), ("valid.md", "日本語\n\t".encode()),
    ]
    corpus = policy.build_corpus(files)
    assert set(corpus) == {"limit.txt", "valid.md"}
    assert corpus["valid.md"] == "日本語\n\t"


def test_artifact_set_excludes_policy_and_allowlist():
    assert policy.artifact_paths([
        "results/ARTIFACT_POLICY.md", policy.ALLOWLIST, "docs/file.md",
        "results_other/a.csv", "experiments/results/a.csv", "results/a.csv", "results/a.csv",
    ]) == ["experiments/results/a.csv", "results/a.csv"]


def test_reference_cannot_span_files():
    artifact = "results/fixture_alpha.csv"
    assert classify([artifact], [("a.md", b"results/fixture_"), ("b.md", b"alpha.csv")]) == ([], [artifact])
