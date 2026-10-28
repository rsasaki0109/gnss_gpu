"""CPU GPS C1 baseline on public RINEX, compared with a separate RTKLIB build.

This checks real-observation integration, not weather effects or truth accuracy.
The observation header approximate coordinate is initialization/context only.
"""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import (read_nav_rinex, read_gps_klobuchar_from_nav_header,
                                  _datetime_to_gps_seconds_of_week)
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.positioning_cpu import solve_pseudorange_position
from gnss_gpu.position_uncertainty_cpu import solve_weighted_pseudorange_position
from gnss_gpu.spp import (_ecef_to_llh, _elevation_azimuth, _tropo_saastamoinen,
                          _iono_klobuchar)
from gnss_gpu.transmission_time import C_LIGHT, sagnac_rotate


CONFIG = """pos1-posmode =single
pos1-frequency =l1
pos1-elmask =15
pos1-ionoopt =brdc
pos1-tropopt =saas
pos1-sateph =brdc
pos1-navsys =1
pos2-rejgdop =30
out-solformat =xyz
out-timesys =gpst
out-timeform =hms
out-timendec =3
"""


def fetch_public_inputs(data, registry):
    """Download the small pinned public inputs, preserving original notices."""
    import requests

    data, registry = Path(data), Path(registry)
    source = json.loads(registry.read_text())
    data.mkdir(parents=True, exist_ok=True)
    for name, info in source["files"].items():
        target = data/Path(name).name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"existing input differs from pinned source: {target}")
            continue
        # Inherited HTTPS proxy and CA trust are retained by requests.
        with requests.get(info["url"], timeout=30, stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > info["bytes"]:
                    raise ValueError("download exceeds pinned input size")
        if len(content) != info["bytes"] or hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError("download differs from pinned public input")
        target.write_bytes(content)
    (data/"source_manifest.json").write_text(json.dumps(source, indent=2)+"\n")


def corrected_epoch(epoch, eph, receiver_approx, alpha, beta):
    """C1 only; broadcast clock/TGD, transmission epoch, Sagnac and atmosphere.

    Use real C1 observations only: never fill missing ranges or C/N0. Satellite
    health, ephemeris age and a fixed 15 degree elevation mask are observables.
    No reference trajectory or weather field is accepted by this function.
    """
    tow = _datetime_to_gps_seconds_of_week(epoch.time)
    lat, lon, alt = _ecef_to_llh(*receiver_approx)
    satellites, ranges, sigmas, ids = [], [], [], []
    for sid in epoch.satellites:
        observation = epoch.observations.get(sid, {})
        pr = observation.get("C1", 0.)
        if not sid.startswith("G") or not np.isfinite(pr) or not 1e6 < pr < 1e8:
            continue
        nav = eph.select_ephemeris(sid, tow)
        if nav is None or nav.sv_health != 0:
            continue
        age = abs((tow-nav.toe+302400.) % 604800.-302400.)
        if age > 7200.:
            continue
        # RINEX epoch is receiver time; P/c removes its common clock offset.
        # Correct satellite clock before recomputing orbital position.
        tx = tow-pr/C_LIGHT
        # Orbital transmit time uses clock without code group delay.
        _, orbital_clock = eph._compute_single_cpu(nav, tx, "C1", apply_group_delay=False)
        sat, clock = eph._compute_single_cpu(nav, tx-orbital_clock, "C1")
        travel = np.linalg.norm(sat-receiver_approx)/C_LIGHT
        sat = sagnac_rotate(sat, travel)
        elevation, azimuth = _elevation_azimuth(receiver_approx, sat)
        if elevation < np.deg2rad(15.):
            continue
        corrected = pr+C_LIGHT*clock
        corrected -= _tropo_saastamoinen(lat, alt, elevation)
        corrected -= _iono_klobuchar(alpha, beta, lat, lon, azimuth, elevation, tow)
        satellites.append(sat); ranges.append(corrected)
        # Relative elevation weighting only; no claimed calibrated covariance.
        sigmas.append(1./np.sin(elevation)); ids.append(sid)
    return np.asarray(satellites).reshape(-1, 3), np.asarray(ranges), np.asarray(sigmas), ids


def read_rtklib_positions(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        fields = line.split()
        timestamp = datetime.strptime(" ".join(fields[:2]), "%Y/%m/%d %H:%M:%S.%f")
        if timestamp in result:
            raise ValueError("duplicate RTKLIB timestamp")
        result[timestamp] = np.array([float(x) for x in fields[2:5]])
    return result


def solution_gpst(observation_time, clock_bias_m):
    """RTKLIB pntpos solution time removes receiver clock from RINEX time.

    Quantize to the configured 1 ms output precision, including second rollover.
    This is a clock-model conversion, not nearest-trajectory/best-offset matching.
    """
    if not np.isfinite(clock_bias_m):
        raise ValueError("finite receiver clock required")
    physical_time = observation_time-timedelta(seconds=float(clock_bias_m)/C_LIGHT)
    return physical_time.replace(microsecond=0)+timedelta(
        milliseconds=(physical_time.microsecond+500)//1000)


def geometric_dop(satellites, position):
    delta = position-satellites
    design = np.column_stack([delta/np.linalg.norm(delta, axis=1)[:, None], np.ones(len(delta))])
    singular = np.linalg.svd(design, compute_uv=False)
    return float(np.sqrt(np.sum(1./singular**2)))


def plot_results(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True)
    start = datetime.fromisoformat(rows[0]["observation_time_gpst"])
    for method in ("equal", "elevation_weighted"):
        subset = [r for r in rows if r["method"] == method and r["rtklib_difference_m"] is not None]
        axes[0].plot([(datetime.fromisoformat(r["observation_time_gpst"])-start).total_seconds()/60 for r in subset],
                     [r["rtklib_difference_m"] for r in subset], label=method)
    equal = [r for r in rows if r["method"] == "equal"]
    t = [(datetime.fromisoformat(r["observation_time_gpst"])-start).total_seconds()/60 for r in equal]
    axes[1].plot(t, [r["gdop"] for r in equal], label="CPU geometry")
    axes[1].axhline(30, color="red", linestyle="--", label="RTKLIB GDOP limit")
    axes[0].set(ylabel="3-D difference from RTKLIB (m)", title="Public RINEX: algorithm agreement, not true position error")
    axes[1].set(ylabel="GDOP (dimensionless)", xlabel="Minutes since recording start")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    figure.savefig(output/"public_rinex_baseline.png", dpi=160)
    plt.close(figure)


def run(data, output, binary):
    data, output, binary = Path(data), Path(output), Path(binary).resolve()
    output.mkdir(parents=True, exist_ok=True)
    obs_path, nav_path = data/"07590920.05o", data/"07590920.05n"
    config_path = output/"rtklib.conf"
    config_path.write_text(CONFIG)
    public_source = json.loads((data/"source_manifest.json").read_text())
    for path in (obs_path, nav_path):
        if hashlib.sha256(path.read_bytes()).hexdigest() != public_source["files"]["test/data/rinex/"+path.name]["sha256"]:
            raise ValueError("input differs from downloaded public-source manifest")
    provenance = dict(
        purpose="Real-observation pipeline check; not rain evaluation or superiority/accuracy proof",
        observation_sha256=hashlib.sha256(obs_path.read_bytes()).hexdigest(),
        navigation_sha256=hashlib.sha256(nav_path.read_bytes()).hexdigest(),
        rtklib_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        public_source=public_source,
        policy=dict(observation="GPS C1", elevation_mask_deg=15, max_toe_age_s=7200,
                    sv_health=0, weighting="equal or 1/sin(elevation) relative sigma",
                    initialization="RINEX approximate coordinate; not independent truth"))
    (output/"frozen_inputs.json").write_text(json.dumps(provenance, indent=2)+"\n")
    result_path = output/"rtklib_single.pos"
    command = [str(binary), "-k", str(config_path.resolve()), "-x", "3", "-o", str(result_path.resolve()),
               str(obs_path.resolve()), str(nav_path.resolve())]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    provenance["rtklib_wall_s"] = time.perf_counter()-started
    (output/"rtklib.log").write_text(completed.stdout+completed.stderr)
    reference = read_rtklib_positions(result_path)
    trace = Path(str(result_path)+".trace")
    provenance["rtklib_rejection_trace_lines"] = [line for line in trace.read_text().splitlines()
                                                if "point pos error" in line]
    obs = read_rinex_obs(obs_path)
    eph = Ephemeris(read_nav_rinex(nav_path, key_by_sat_id=True))
    alpha, beta = read_gps_klobuchar_from_nav_header(nav_path)
    if alpha is None or beta is None:
        raise ValueError("broadcast ionosphere coefficients required, no silent defaults")
    initial = obs.header.approx_position.copy()
    if not np.all(np.isfinite(initial)) or not 6e6 < np.linalg.norm(initial) < 7e6:
        raise ValueError("valid approximate ECEF coordinate required")
    rows = []
    started = time.perf_counter()
    for epoch in obs.epochs:
        sat, ranges, sigma, ids = corrected_epoch(epoch, eph, initial, alpha, beta)
        weighted = solve_weighted_pseudorange_position(sat, ranges, sigma, initial_position_m=initial)
        solutions = dict(equal=solve_pseudorange_position(sat, ranges, initial_position_m=initial),
                         elevation_weighted=weighted.solution if weighted is not None else None)
        for method, solution in solutions.items():
            corrected_time = solution_gpst(epoch.time, solution.clock_bias_m) if solution else None
            baseline = reference.get(corrected_time)
            rows.append(dict(observation_time_gpst=epoch.time.isoformat(),
                             solution_time_gpst=corrected_time.isoformat() if corrected_time else None,
                             method=method, input_satellites=len(epoch.satellites),
                             used_satellites=len(ids), used_ids=";".join(ids), available=solution is not None,
                             rtklib_available=baseline is not None,
                             x_m=solution.position_m[0] if solution else None,
                             y_m=solution.position_m[1] if solution else None,
                             z_m=solution.position_m[2] if solution else None,
                             receiver_clock_bias_m=solution.clock_bias_m if solution else None,
                             gdop=geometric_dop(sat, solution.position_m) if solution else None,
                             residual_rms_m=float(np.sqrt(np.mean(solution.residuals_m**2))) if solution else None,
                             rtklib_difference_m=float(np.linalg.norm(solution.position_m-baseline))
                             if solution is not None and baseline is not None else None,
                             header_offset_m=float(np.linalg.norm(solution.position_m-initial)) if solution else None))
    provenance["python_pipeline_wall_s"] = time.perf_counter()-started
    with (output/"positions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    provenance["results"] = dict(input_epochs=len(obs.epochs), rtklib_positions=len(reference),
                                  nominal_interval_s=obs.header.interval,
                                  cn0_observation_present=any(c.startswith("S") for c in obs.header.obs_types["G"]))
    for method in ("equal", "elevation_weighted"):
        subset = [r for r in rows if r["method"] == method]
        paired = [r["rtklib_difference_m"] for r in subset if r["rtklib_difference_m"] is not None]
        matched_times = [r["solution_time_gpst"] for r in subset if r["rtklib_difference_m"] is not None]
        if not paired or len(set(matched_times)) != len(matched_times):
            raise RuntimeError("comparison requires nonempty, unique clock-corrected epoch matches")
        provenance["results"][method] = dict(available=sum(r["available"] for r in subset), paired=len(paired),
            rtklib_difference_rmse_m=float(np.sqrt(np.mean(np.square(paired)))) if paired else None,
            rtklib_difference_max_m=max(paired) if paired else None)
    provenance["limitations"] = [
        "No independent reference coordinate/time series; differences from RTKLIB are not true errors.",
        "No precipitation, wetness or forest labels; not a weather dataset.",
        "C/N0 and receiver acquisition/tracking states absent; cannot validate RF recovery.",
        "Same input/nav does not imply identical estimator: RTKLIB adds its own variance and screening models.",
        "120 correlated epochs are one development recording, not 120 independent experiments.",
        "Wall times cover different scopes/languages and are not a speed comparison."]
    plot_results(output, rows)
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance["results"], indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("output/public-gnss-data/rtklib"))
    parser.add_argument("--output", type=Path, default=Path("output/public-rinex-baseline"))
    parser.add_argument("--rtklib-binary", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="fetch checksum-pinned inputs before evaluation")
    parser.add_argument("--sources", type=Path, default=Path("docs/public_rinex_baseline_sources.json"))
    args = parser.parse_args()
    if args.fetch:
        fetch_public_inputs(args.data, args.sources)
    run(args.data, args.output, args.rtklib_binary)
