"""Annotate saved receiver traces with carrier search-boundary diagnostics.

This reuses saved estimates, not receiver execution or signal truth, for the
diagnostic. Truth labels are used only for subsequent error summaries.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gnss_gpu.acquisition_cpu import CaAcquisition, CaSearch
import gnss_gpu.carrier_support_cpu as support
from gnss_gpu.code_delay_cpu import CodeDelayEstimate


def decode_complex(value):
    return complex(value["real"], value["imag"]) if set(value) == {"real", "imag"} else value


def audit(recording_dir, output, *, boundary_margin_hz=.1):
    recording_dir, output = Path(recording_dir), Path(output)
    metadata_path, traces_path = recording_dir/"metadata.json", recording_dir/"traces.jsonl"
    metadata = json.loads(metadata_path.read_text())
    search = CaSearch(**metadata["threshold"]["search"])
    # Validate configuration even if a recording contains no candidates.
    support.assess_ca_carrier_support(None, search, boundary_margin_hz=boundary_margin_hz)
    summaries, diagnostics, seen = [], [], set()
    for line in traces_path.read_text().splitlines():
        record = json.loads(line, object_hook=decode_complex)
        condition = str(record["condition"])
        if condition in seen or condition not in metadata["iq_sha256"]:
            raise ValueError("duplicate or unknown scenario")
        seen.add(condition)
        if len(record["trace"]) != metadata["blocks"]:
            raise ValueError("incomplete scenario trace")
        counts, annotated = Counter(), []
        boundary_errors, interior_errors = [], []
        for row in record["trace"]:
            data = row["refined_candidate"]
            candidate = None if data is None else CaAcquisition(data["carrier_hz"],
                        CodeDelayEstimate(**data["delay"]), data["statistic"])
            assessment = support.assess_ca_carrier_support(candidate, search, boundary_margin_hz=boundary_margin_hz)
            if row["measurement_used"]:
                if assessment.status == "not_estimated":
                    raise ValueError("fresh measurement lacks a refined candidate")
                counts[assessment.status] += 1
                error = abs(row["carrier_hz"]-record["primary_doppler_hz"])
                if assessment.status == "boundary":
                    boundary_errors.append(error)
                elif assessment.status == "interior":
                    interior_errors.append(error)
            annotated.append(dict(start_sample=row["start_sample"], receiver_status=row["status"],
                                  measurement_used=row["measurement_used"], support=asdict(assessment)))
        fields = ("condition", "nominal_primary_cn0_db_hz", "primary_doppler_hz", "outage_ms", "replica")
        identity = {key: record[key] for key in fields}
        summaries.append(dict(**identity, used_blocks=sum(counts.values()),
            boundary_used_blocks=counts["boundary"], interior_used_blocks=counts["interior"],
            fixed_frequency_used_blocks=counts["fixed_frequency"],
            boundary_conditional_carrier_mae_hz=float(np.mean(boundary_errors)) if boundary_errors else None,
            interior_conditional_carrier_mae_hz=float(np.mean(interior_errors)) if interior_errors else None))
        diagnostics.append(dict(**identity, blocks=annotated))
    if seen != set(metadata["iq_sha256"]):
        raise ValueError("missing scenarios from recording")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(scenarios=len(summaries), blocks=sum(len(record["blocks"]) for record in diagnostics),
                  boundary_margin_hz=boundary_margin_hz, search=asdict(search),
                  used_blocks=sum(row["used_blocks"] for row in summaries),
                  boundary_used_blocks=sum(row["boundary_used_blocks"] for row in summaries),
                  source_input_sha256={"metadata.json": digest(metadata_path), "traces.jsonl": digest(traces_path)},
                  source_sha256={"audit": digest(Path(__file__)), "diagnostic": digest(Path(support.__file__))},
                  diagnostic_uses_truth=False, receiver_reexecuted=False,
                  limitations=["Boundary proximity is descriptive, not proof the true carrier lies outside the grid.",
                               "True edge signals also trigger; interior estimates can be incorrect.",
                               "No receiver gates, state updates, grid or frozen trace were changed.",
                               "Conditional errors and this small synthetic matrix do not establish integrity or general sensitivity."])
    output.mkdir(parents=True, exist_ok=True)
    (output/"diagnostics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in diagnostics))
    (output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, default=Path("output/cpu-receiver-stress"))
    parser.add_argument("--output", type=Path, default=Path("output/cpu-carrier-support"))
    parser.add_argument("--boundary-margin-hz", type=float, default=.1)
    args = parser.parse_args()
    report, _ = audit(args.recording_dir, args.output, boundary_margin_hz=args.boundary_margin_hz)
    print(json.dumps(report, indent=2))
