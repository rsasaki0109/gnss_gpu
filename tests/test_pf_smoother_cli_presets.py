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


def test_print_cli_presets_lists_available_presets(capsys):
    print_cli_presets()

    out = capsys.readouterr().out
    assert "Available presets:" in out
    assert "odaiba_reference:" in out
    assert set(CLI_PRESETS) >= {"odaiba_reference", "odaiba_best_accuracy"}
