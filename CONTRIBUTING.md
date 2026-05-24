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
python3 -m pip install pytest ruff pandas scipy requests matplotlib plotly
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

## Before you open a pull request

- **Tests**: add or update tests under `tests/` and run
  `PYTHONPATH=python python3 -m pytest tests/ -q`. Tests that need the native
  kernels may be skipped locally; that's fine.
- **Lint**: run `ruff check .` (CI runs the same).
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

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
