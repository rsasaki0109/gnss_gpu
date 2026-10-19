"""Capture every D236 actual likelihood call, leaving its original result intact."""

import argparse
import gzip
import json
import os
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_irls_replay import snapshot, write_record
from rbpf_likelihood_batch import BatchScorer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--parent-extension", type=Path, required=True)
    parser.add_argument("--mass-extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    own, remaining = parser.parse_known_args()
    path = own.output_dir.with_suffix(".corpus.jsonl.gz")
    assert not path.exists()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    configs, indices = [], []
    prior_step, prior_scorer = field.MoveBatchStep, field.BatchScorer
    with gzip.open(path, "wt", encoding="utf8", newline="\n") as stream:

        class CaptureScorer(BatchScorer):
            def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
                saved = snapshot(hypotheses, problem, oracle)
                result = super().score(hypotheses, problem, oracle, mode, diagnostics)
                write_record(stream, len(indices), saved, result)
                indices.append(len(indices))
                return result

        def select(runtime, scorer, mode):
            assert runtime.cfg.cluster_relin == own.cluster_relin
            configs.append(vars(runtime.cfg).copy())
            return NativeFeedbackMassStep(
                runtime,
                scorer,
                mode,
                extension=own.extension,
                parent_extension=own.parent_extension,
                mass_extension=own.mass_extension,
            )

        field.MoveBatchStep, field.BatchScorer = select, CaptureScorer
        sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
        try:
            code = field.main()
        finally:
            field.MoveBatchStep, field.BatchScorer = prior_step, prior_scorer
    (own.output_dir / "capture.json").write_text(
        json.dumps(
            dict(
                configs=configs,
                batches=len(indices),
                corpus_sha256=field.sha(path),
                exit_code=code,
                timing_comparable=False,
                capture_inside_step_timer=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
