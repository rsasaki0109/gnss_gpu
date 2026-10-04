import pytest

from gnss_gpu.pf_smoother_cli_presets import (
    CLI_PRESETS,
    expand_cli_preset_argv,
    print_cli_presets,
)


def test_expand_cli_preset_argv_expands_known_preset_and_keeps_late_flags():
    expanded = expand_cli_preset_argv(
        ["--preset", "odaiba_reference", "--sigma-pos", "2.0"]
    )

    assert "--preset" not in expanded
    assert expanded[-2:] == ["--sigma-pos", "2.0"]
    assert "--smoother" in expanded
    assert "--no-doppler-per-particle" in expanded


def test_expand_cli_preset_argv_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset 'missing'"):
        expand_cli_preset_argv(["--preset=missing"])


def test_expand_cli_preset_argv_expands_odaiba_pf_nlos_soft():
    expanded = expand_cli_preset_argv(["--preset", "odaiba_pf_nlos_soft"])
    assert "--nlos-k-weak" in expanded
    assert "3.0" in expanded
    assert "--smoother" in expanded


def test_print_cli_presets_lists_available_presets(capsys):
    print_cli_presets()

    out = capsys.readouterr().out
    assert "Available presets:" in out
    assert "odaiba_reference:" in out
    assert set(CLI_PRESETS) >= {"odaiba_reference", "odaiba_best_accuracy", "odaiba_pf_nlos_soft"}


def test_urbannav_rtk_anchored_preset_extends_stop_detect():
    expanded = expand_cli_preset_argv(
        ["--preset", "urbannav_rtk_anchored", "--rtk-anchor-pos", "rtk.pos"]
    )
    stop_detect = CLI_PRESETS["odaiba_stop_detect"]["argv"]
    assert expanded[: len(stop_detect)] == stop_detect
    assert "--rtk-anchor-heading" in expanded
    # Later --sigma-pos wins over the 1.2 m inherited from odaiba_stop_detect.
    assert expanded[len(expanded) - 1 - expanded[::-1].index("--sigma-pos") + 1] == "0.1"
    assert expanded[-2:] == ["--rtk-anchor-pos", "rtk.pos"]


def test_rtk_anchored_doppler_preset_extends_urbannav_rtk_anchored():
    expanded = expand_cli_preset_argv(["--preset", "rtk_anchored_doppler"])
    base = CLI_PRESETS["urbannav_rtk_anchored"]["argv"]
    assert expanded[: len(base)] == base
    assert expanded[len(base):] == ["--imu-speed-source", "doppler", "--imu-gyro-bias-zupt"]
