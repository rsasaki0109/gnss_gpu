"""Actual accuracy prefix,counting the captured instance-bound candidate score."""

import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_cluster_parameter_reuse import ClusterBatchScorer


def main():
    proxy = SimpleNamespace(**vars(field))
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    created = []

    class SelectedScorer(ClusterBatchScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, extension=os.environ['GNSS_GPU_RBPF_CLUSTER_BATCH_EXT'], **kwargs)
            # The candidate constructor binds score on the instance. Capture
            # that exact method before adding the outer diagnostic wrapper.
            self.numeric_score = self.score
            self.score = self._counted_score
            self.actual_calls = dict(gpu=0, native_cpu=0)
            created.append(self)

        def _counted_score(self, hypotheses, problem, oracle, mode='gpu', diagnostics=False):
            self.actual_calls[mode] += 1
            values, detail = self.numeric_score(hypotheses, problem, oracle, mode, diagnostics=True)
            assert detail['scalar_fallbacks'] == 0 and 'diagnostic' in detail
            if not diagnostics:
                detail = {k:v for k,v in detail.items() if k not in ('diagnostic', 'unique_hypotheses')}
            return values, detail

    def selected_field():
        original = field.main
        run = FunctionType(original.__code__, dict(original.__globals__, BatchScorer=SelectedScorer,
            MoveBatchStep=proxy.MoveBatchStep), original.__name__, original.__defaults__, original.__closure__)
        return run()

    proxy.main = selected_field
    run = FunctionType(base.main.__code__, dict(base.main.__globals__, field=proxy),
        base.main.__name__, base.main.__defaults__, base.main.__closure__)
    code = run()
    assert len(created) == 1 and created[0].actual_calls == dict(gpu=197, native_cpu=0)
    scorer = created[0]
    assert scorer.kernel == 'register' and scorer.numeric_score.__self__ is scorer
    ci = int(os.environ['RBPF_CLUSTER_RELIN'])
    counts = scorer.cluster_engine.counts()
    assert counts['batch_calls'] == 197*ci and counts['fallback_calls'] == 197*(1-ci)
    (output/'transport.json').write_text(json.dumps(dict(kernel='register', actual_score_calls=scorer.actual_calls,
        diagnostic_only=True, original_modules_not_mutated=True, accuracy_only_full_diagnostics=True,
        cluster_batch=counts, old_engine_counters_are_logical_requests=True,
        captured_instance_candidate_score=True),indent=2)+'\n',encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
