"""CLI presets for PF smoother experiments."""

from __future__ import annotations

CLI_PRESETS: dict[str, dict[str, object]] = {
    "odaiba_reference": {
        "description": "Smoother-first Odaiba reference: IMU stop-detect plus 0.18-cycle DD floor.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
        ],
    },
    "odaiba_stop_detect": {
        "description": "Forward-stable Odaiba sibling: IMU stop-detect plus 0.25-cycle DD floor.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.25",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
        ],
    },
    "odaiba_reference_guarded": {
        "description": "Odaiba reference + low-ESS smoother tail guard for weak smoothing tails.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
            "--smoother-tail-guard-ess-max-ratio", "0.001",
            "--smoother-tail-guard-min-shift-m", "4.0",
        ],
    },
    "odaiba_best_accuracy": {
        "description": "Odaiba P50-minimizing config (200K + auto stop constant + anchor-sigma 0.15).",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "200000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.15",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
            "--stop-segment-constant",
            "--stop-segment-source", "smoothed_auto_tail",
            "--stop-segment-density-neighbors", "200",
            "--smoother-tail-guard-ess-max-ratio", "0.0001",
            "--smoother-tail-guard-min-shift-m", "9.0",
            "--smoother-tail-guard-expand-epochs", "10",
            "--smoother-tail-guard-expand-dd-pseudorange-max-pairs", "0",
        ],
    },
}

for _preset_name in (
    "odaiba_reference",
    "odaiba_stop_detect",
    "odaiba_reference_guarded",
    "odaiba_best_accuracy",
):
    CLI_PRESETS[_preset_name]["argv"].append("--no-doppler-per-particle")

CLI_PRESETS["odaiba_rbpf_velocity"] = {
    "description": "Experimental Odaiba proper RBPF velocity KF probe.",
    "argv": [
        *CLI_PRESETS["odaiba_best_accuracy"]["argv"],
        "--rbpf-velocity-kf",
        "--rbpf-velocity-init-sigma", "2.0",
        "--rbpf-velocity-process-noise", "1.0",
        "--rbpf-doppler-sigma", "0.5",
        "--doppler-min-sats", "4",
        "--pf-sigma-vel", "0.0",
        "--pf-velocity-guide-alpha", "1.0",
        "--pf-init-spread-vel", "0.0",
    ],
}


def expand_cli_preset_argv(argv: list[str]) -> list[str]:
    """Inline preset argv fragments so later user flags keep normal precedence."""

    expanded: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--preset":
            if i + 1 >= len(argv):
                raise ValueError("--preset requires a preset name")
            preset_name = argv[i + 1]
            expanded.extend(_preset_argv(preset_name))
            i += 2
            continue
        if token.startswith("--preset="):
            preset_name = token.split("=", 1)[1]
            expanded.extend(_preset_argv(preset_name))
            i += 1
            continue
        if token == "--list-presets":
            i += 1
            continue
        expanded.append(token)
        i += 1
    return expanded


def print_cli_presets() -> None:
    print("Available presets:")
    for name in sorted(CLI_PRESETS):
        print(f"  {name}: {CLI_PRESETS[name]['description']}")


def _preset_argv(preset_name: str) -> list[str]:
    preset = CLI_PRESETS.get(preset_name)
    if preset is None:
        known = ", ".join(sorted(CLI_PRESETS))
        raise ValueError(f"unknown preset '{preset_name}' (known: {known})")
    return list(preset["argv"])
