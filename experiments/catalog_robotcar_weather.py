"""Inventory official RobotCar weather labels without selecting on GPS outcome.

Only metadata is inventoried. A rain label is not a measured rain rate, antenna
wetness, or evidence of rain causing a positioning error.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def catalog(sdk, output):
    sdk, output = Path(sdk), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, counts, hashes = [], Counter(), {}
    for path in sorted((sdk/"tags").glob("*.yaml")):
        # Official files contain a flat YAML list of literal tag names.
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if any(not line.startswith("- ") for line in lines):
            raise ValueError(f"unsupported metadata format: {path}")
        tags = [line[2:] for line in lines]
        counts.update(set(tags))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(sequence=path.stem, tags=";".join(tags),
                         rain="rain" in tags, sun="sun" in tags,
                         gps_quality_label=";".join(t for t in tags if t in ("poor-gps", "no-gps")),
                         source=f"https://github.com/ori-mrg/robotcar-dataset-sdk/blob/"))
    if not rows:
        raise ValueError("no official weather tag files found")
    commit = subprocess.check_output(["git", "-C", str(sdk), "rev-parse", "HEAD"], text=True).strip()
    for row in rows:
        row["source"] += f"{commit}/tags/{row['sequence']}.yaml"
    with (output/"robotcar_weather_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(repository="https://github.com/ori-mrg/robotcar-dataset-sdk", commit=commit,
                   sequences=len(rows), tag_counts=dict(sorted(counts.items())),
                   rain_sequences=[r["sequence"] for r in rows if r["rain"]],
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   tag_file_sha256=hashes,
                   scope="Official metadata only; GNSS payload not downloaded or evaluated.",
                   selection_rule="Inventory every tag file; do not exclude poor-gps/no-gps based on outcomes.",
                   warning="Sun is a label, not verified dry antenna. Tags do not measure rain rate or wetness.")
    (output/"robotcar_weather_provenance.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("commit", "sequences", "tag_counts", "rain_sequences")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/public-gnss-data"))
    args = parser.parse_args()
    catalog(args.sdk, args.output)
