from __future__ import annotations

import csv
import json

from experiments.audit_gsdc2023_matlab_submission_score_equivalence import (
    audit_matlab_submission_score_equivalence,
    main,
)


def _write_submission(path, lat_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n"
        f"trip/a,1000,{37.0 + lat_offset},-122.0\n"
        f"trip/a,2000,{37.1 + lat_offset},-122.1\n",
        encoding="utf-8",
    )


def test_audit_matlab_submission_score_equivalence_marks_identical_and_scores(tmp_path) -> None:
    reference = tmp_path / "reference" / "submission_matlab.csv"
    identical = tmp_path / "candidates" / "submission_matlab.csv"
    shifted = tmp_path / "candidates" / "submission_shifted.csv"
    _write_submission(reference)
    _write_submission(identical)
    _write_submission(shifted, lat_offset=0.00001)
    score_log = tmp_path / "scores.csv"
    score_log.write_text(
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission_shifted.csv,now,shifted,complete,1.23,4.56\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.csv"

    rows = audit_matlab_submission_score_equivalence(
        matlab_reference=reference,
        candidate_roots=[tmp_path / "candidates"],
        submitted_csvs=[score_log],
        output_csv=output,
    )

    assert rows[0]["filename"] == "submission_matlab.csv"
    assert rows[0]["byte_identical"] is True
    shifted_row = next(row for row in rows if row["filename"] == "submission_shifted.csv")
    assert shifted_row["byte_identical"] is False
    assert shifted_row["changed_rows"] == 2
    assert shifted_row["score_log_public"] == "1.23"
    assert output.is_file()
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["byte_identical_count"] == 1
    assert summary["closest"]["filename"] == "submission_matlab.csv"


def test_audit_matlab_submission_score_equivalence_cli(tmp_path, capsys) -> None:
    reference = tmp_path / "reference" / "submission_matlab.csv"
    candidate = tmp_path / "candidate" / "submission_candidate.csv"
    _write_submission(reference)
    _write_submission(candidate)
    output = tmp_path / "audit.csv"

    assert (
        main(
            [
                "--matlab-reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--output-csv",
                str(output),
            ],
        )
        == 0
    )
    assert "audited: 1 candidate(s)" in capsys.readouterr().out
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["byte_identical"] == "True"
    assert rows[0]["changed_rows"] == "0"
