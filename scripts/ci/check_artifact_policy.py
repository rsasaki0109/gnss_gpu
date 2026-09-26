"""Check that tracked result artifacts have external references or an exception."""

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
import subprocess

RESULT_ROOTS = ("results/", "experiments/results/")
ALLOWLIST = "results/artifact_allowlist.txt"
EXCLUDED = {"results/ARTIFACT_POLICY.md", ALLOWLIST}
MAX_TEXT_BYTES = 2 * 1024 * 1024
CODE_SUFFIXES = {".py", ".sh", ".yml"}


def result_root(path: str) -> str | None:
    """Return the containing result root, including its trailing slash."""
    return next((root for root in RESULT_ROOTS if path.startswith(root)), None)


def artifact_paths(tracked: Iterable[str]) -> list[str]:
    return sorted({path for path in tracked if result_root(path) and path not in EXCLUDED})


def corpus_text(path: str, data: bytes) -> str | None:
    """Filter one tracked file; result files, binaries and large files cannot cite."""
    if result_root(path) or len(data) > MAX_TEXT_BYTES:
        return None
    # NUL and non-text ASCII controls identify binaries even if UTF-8 decodes.
    if re.search(rb"[\x00-\x08\x0b\x0e-\x1f]", data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def build_corpus(files: Iterable[tuple[str, bytes]]) -> dict[str, str]:
    """Build the reference corpus from tracked (path, contents) pairs."""
    corpus = {}
    for path, data in files:
        text = corpus_text(path, data)
        if text is not None:
            corpus[path] = text
    return corpus


def parse_allowlist(text: str) -> tuple[str, ...]:
    """Read case-sensitive repo-relative fnmatch globs, with # comments."""
    return tuple(line for raw in text.splitlines() if (line := raw.split("#", 1)[0].strip()))


def ancestors(path: str) -> tuple[str, ...]:
    """Directories strictly below the result root, outermost first."""
    root = result_root(path)
    if root is None:
        return ()
    parts = path[len(root):].split("/")
    return tuple(root + "/".join(parts[:i]) for i in range(1, len(parts)))


def workspace(path: str) -> str:
    parents = ancestors(path)
    return parents[0] if parents else (result_root(path) or "") + "(root)"


def classify_artifacts(
    artifacts: Iterable[str], corpus: Mapping[str, str], allowlist: Iterable[str] = ()
) -> tuple[list[str], list[str]]:
    """Partition artifacts using exact paths, ancestors, code names, and globs."""
    artifacts = sorted(set(artifacts))
    patterns = tuple(allowlist)
    # The separator prevents a path from spanning two separate source files.
    joined = "\n".join(corpus.values())
    names = {PurePosixPath(parents[0]).name for a in artifacts if (parents := ancestors(a))}
    code_names = set()
    if names:
        word_pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        for path, text in corpus.items():
            if PurePosixPath(path).suffix in CODE_SUFFIXES and "results" in text:
                code_names.update(word_pattern.findall(text))

    @lru_cache(maxsize=None)
    def directory_referenced(directory: str) -> bool:
        return re.search(re.escape(directory) + r"(?=[/\"'\s)\x60]|$)", joined) is not None

    referenced, unreferenced = [], []
    for artifact in artifacts:
        parents = ancestors(artifact)
        kept = (
            any(fnmatchcase(artifact, pattern) for pattern in patterns)
            or (bool(parents) and PurePosixPath(parents[0]).name in code_names)
            or any(directory_referenced(parent) for parent in parents)
            or artifact in joined
        )
        (referenced if kept else unreferenced).append(artifact)
    return referenced, unreferenced


def tracked_files(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [path.decode("utf-8") for path in output.split(b"\0") if path]


def read_corpus(root: Path, tracked: Iterable[str]) -> dict[str, str]:
    def contents():
        for path in tracked:
            if result_root(path):
                continue
            source = root / path
            # Submodules are directories; missing tracked files cannot cite.
            if source.is_file() and source.stat().st_size <= MAX_TEXT_BYTES:
                yield path, source.read_bytes()

    return build_corpus(contents())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-unreferenced", action="store_true", help="print only unreferenced paths")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    tracked = tracked_files(root)
    allowlist_path = root / ALLOWLIST
    allowlist = parse_allowlist(allowlist_path.read_text(encoding="utf-8"))
    referenced, unreferenced = classify_artifacts(
        artifact_paths(tracked), read_corpus(root, tracked), allowlist
    )
    if args.list_unreferenced:
        for path in unreferenced:
            print(path)
    else:
        print(f"Artifact policy: {len(referenced)} referenced, {len(unreferenced)} unreferenced")
        kept = Counter(map(workspace, referenced))
        missing = Counter(map(workspace, unreferenced))
        for name in sorted(kept.keys() | missing.keys()):
            print(f"  {name}: {kept[name]} referenced, {missing[name]} unreferenced")
    return int(bool(unreferenced))


if __name__ == "__main__":
    raise SystemExit(main())
