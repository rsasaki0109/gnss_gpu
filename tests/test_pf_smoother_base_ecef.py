"""--base-ecef reaches every base-relative observation computer."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from gnss_gpu.pf_smoother_cli_config import namespace_to_run_config
from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser
from gnss_gpu.pf_smoother_runtime import build_observation_computers

BASE = (-3961903.71, 3348994.03, 3698211.68)


def _config(extra: list[str]):
    parser = build_pf_smoother_arg_parser(1.2)
    args = parser.parse_args(
        ["--data-root", "unused", "--dd-pseudorange", "--widelane", "--mupf-dd", *extra]
    )
    return namespace_to_run_config(args, position_update_sigma=1.9, use_smoother=False)


def test_default_keeps_rinex_header_base():
    assert _config([]).observations.base_ecef is None


def test_base_ecef_flows_into_observation_config():
    cfg = _config(["--base-ecef", *map(str, BASE)])
    assert cfg.observations.base_ecef == BASE


def test_build_observation_computers_passes_base_position(tmp_path: Path, monkeypatch):
    (tmp_path / "base_trimble.obs").write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake(name):
        class _Computer:
            def __init__(self, *args, **kwargs):
                base = kwargs.get("base_position")
                seen[name] = None if base is None else tuple(float(v) for v in base)
                self.base_position = base

        return _Computer

    for module, cls in (
        ("gnss_gpu.dd_pseudorange", "DDPseudorangeComputer"),
        ("gnss_gpu.widelane", "WidelaneDDPseudorangeComputer"),
        ("gnss_gpu.dd_carrier", "DDCarrierComputer"),
    ):
        stub = types.ModuleType(module)
        setattr(stub, cls, fake(cls))
        monkeypatch.setitem(sys.modules, module, stub)

    cfg = _config(["--base-ecef", *map(str, BASE)])
    build_observation_computers(tmp_path, "trimble", cfg.observations)

    assert seen == {
        "DDPseudorangeComputer": BASE,
        "WidelaneDDPseudorangeComputer": BASE,
        "DDCarrierComputer": BASE,
    }
