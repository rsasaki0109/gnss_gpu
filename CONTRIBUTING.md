# Contributing to gnss_gpu

Thanks for your interest! `gnss_gpu` is an experiment-first GNSS positioning
workspace, so contributions range from bug fixes in the reusable library to new
positioning experiments. This guide keeps that mix manageable.

## Ways to contribute

- **Report a bug** or **request a feature** via the
  [issue templates](https://github.com/rsasaki0109/gnss_gpu/issues/new/choose).
- **Improve the reusable code** under `python/gnss_gpu/` or the CUDA/C++ kernels
  under `src/`.
- **Add or refine an experiment** under `experiments/`.
- **Improve docs/examples** — especially anything that makes the project easier
  to try (the [`examples/`](examples/) demos are a good place).

## Development setup

No GPU is required for most Python work:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pytest "ruff==0.15.7" pandas scipy requests matplotlib plotly
```

Run the pure-Python smoke demo and the tests before you start:

```bash
PYTHONPATH=python python3 examples/demo_urban_canyon_sim.py
PYTHONPATH=python python3 -m pytest tests/ -q
```

Build the native CUDA/C++ kernels only when your change touches the
GPU-accelerated paths (signal-sim, particle filter, ray tracing, multi-GNSS):

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j"$(nproc)"
# then copy the generated .so files into python/gnss_gpu/
```

On Windows, build from a Visual Studio 2022 x64 developer environment with
Ninja (verified with CUDA 12.8 and Python 3.12). Point CMake at the exact
interpreter you run tests with, or the extensions get the wrong ABI tag
(e.g. `cp314` instead of `cp312`):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
for /f %i in ('python -c "import pybind11;print(pybind11.get_cmake_dir())"') do set PYBIND11_DIR=%i
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -Dpybind11_DIR=%PYBIND11_DIR% -DPython_EXECUTABLE=C:\path\to\python.exe
cmake --build build -j 8
copy build\*.pyd python\gnss_gpu\
```

Replace `75` with your GPU's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`).
A non-editable `gnss-gpu` in site-packages shadows the repo; uninstall it or
use `pip install -e .`.

## Before you open a pull request

- **Tests**: add or update tests under `tests/` and run
  `PYTHONPATH=python python3 -m pytest tests/ -q`. Tests that need the native
  kernels may be skipped locally; that's fine.
- **Lint**: run the same checks as the CI `lint` and `repo-hygiene` jobs
  (`.github/workflows/ci.yml`). CI pins `ruff==0.15.7` and lints only
  `python/` plus an explicit list of PPC promotion-path files; a bare
  `ruff check .` reports hundreds of pre-existing findings under
  `experiments/` and `tests/` that CI does not gate on.

  ```bash
  python -m pip install "ruff==0.15.7"
  python -m ruff check python/ --ignore=E501,F401
  python scripts/ci/check_artifact_policy.py
  python tools/lint_repo_paths.py --max-path 180 --max-name 128
  python tools/lint_commit_messages.py --base origin/main
  ```

  If you touch a file in the PPC promotion-path list in `ci.yml`, also run
  `python -m ruff check <that file>` without the ignores.
- **Type check**: CI runs `python -m pyright` (pinned `pyright==1.1.414`,
  basic mode, config in `pyrightconfig.json`). Files listed under `exclude`
  are pre-existing debt: when you clean one, remove it from the list; never
  add a file to it.
- **Keep PRs focused**: one logical change per PR. Don't bundle unrelated edits.
- **Match the surrounding style**: comment density, naming, and idioms.

## Project conventions

These come from `internal_docs/decisions.md` and the README's development policy:

- Keep stable, reusable code in `python/gnss_gpu/` or `src/`. Keep variant-heavy
  experiment logic in `experiments/` until it survives fixed evaluation.
- **Do not promote a method because it wins one pilot split.** Prefer
  same-input, same-metric comparisons over new abstractions.
- Be honest about results: report failures and skipped steps, not just wins.
- Record durable decisions in `internal_docs/decisions.md`.
- **Do not vendor, link, or derive production code/config from GPL-3.0 reference
  sources** such as `gici-open`. This repo is Apache-2.0.

## Git workflow & branch protection

`main` mirrors `origin/main` — never commit to it directly. On 2026-06-17 a local
`main` diverged into an *unrelated history* (88 local vs 469 remote commits) and a
`git reset --hard` then failed on Windows because tracked sweep filenames exceeded
the 260-char `MAX_PATH` limit. These rules keep that from recurring.

Per-clone setup (run once):

```bash
git config pull.ff only        # never create surprise merge commits on pull
git config fetch.prune true    # drop deleted remote branches
git config push.default simple
git config core.hooksPath .githooks   # enable the tracked pre-push guard
git config core.longpaths true        # tolerate long paths on Windows (belt-and-braces)
```

Day-to-day:

```bash
git fetch --prune origin
git switch main && git reset --hard origin/main   # main is read-only; just track origin
git switch -c feat/<topic>                          # do all work on feature branches
# ... commit ...
git fetch origin && git rebase origin/main          # stay based on origin/main
git push -u origin feat/<topic>                     # then open a PR
```

The tracked `.githooks/pre-push` hook blocks direct pushes to `main` and refuses
to push a branch that isn't a descendant of `origin/main`. Never use
`git pull --allow-unrelated-histories`.

**Commit messages:** do not add `Co-authored-by:` trailers for AI tools (Cursor,
Claude, Copilot, etc.). `.githooks/prepare-commit-msg` strips auto-injected
trailers; `.githooks/commit-msg` rejects any that remain. CI also scans PR
commits via `tools/lint_commit_messages.py`. Human authorship only.

**GitHub settings** (repo admin, Settings → Branches → add rule for `main`):
require a PR before merging, require status checks (`lint`, `repo-hygiene`,
`test-python-smoke`) to pass, require linear history, and disallow force-pushes
and deletions. Apply to administrators too.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
