"""Trip epoch-window helpers shared by GSDC2023 parity tools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FULL_EPOCH_COUNT = 1_000_000_000


def settings_epoch_window_for_trip(trip_dir: Path, requested_max_epochs: int) -> tuple[int, int]:
    trip_dir = Path(trip_dir)
    start_epoch = 0
    setting_max_epochs: int | None = None
    if len(trip_dir.parents) >= 3:
        split = trip_dir.parent.parent.name
        course = trip_dir.parent.name
        phone = trip_dir.name
        data_root = trip_dir.parent.parent.parent
        settings_path = data_root / f"settings_{split}.csv"
        if settings_path.is_file():
            settings = pd.read_csv(settings_path)
            if {"Course", "Phone", "IdxStart", "IdxEnd"}.issubset(settings.columns):
                match = settings[
                    (settings["Course"].astype(str) == str(course))
                    & (settings["Phone"].astype(str) == str(phone))
                ]
                if not match.empty:
                    idx_start = pd.to_numeric(match.iloc[0]["IdxStart"], errors="coerce")
                    idx_end = pd.to_numeric(match.iloc[0]["IdxEnd"], errors="coerce")
                    if pd.notna(idx_start) and int(idx_start) > 0:
                        start_epoch = max(int(idx_start) - 1, 0)
                    if pd.notna(idx_start) and pd.notna(idx_end) and int(idx_end) >= int(idx_start):
                        setting_max_epochs = int(idx_end) - int(idx_start) + 1

    if requested_max_epochs > 0 and setting_max_epochs is not None:
        max_epochs = min(int(requested_max_epochs), setting_max_epochs)
    elif requested_max_epochs > 0:
        max_epochs = int(requested_max_epochs)
    elif setting_max_epochs is not None:
        max_epochs = setting_max_epochs
    else:
        max_epochs = FULL_EPOCH_COUNT
    return start_epoch, max_epochs


def trim_epoch_window(
    frame: pd.DataFrame,
    start_epoch: int,
    max_epochs: int,
    *,
    epoch_column: str = "epoch_index",
    next_epoch_column: str | None = None,
) -> pd.DataFrame:
    if max_epochs >= FULL_EPOCH_COUNT:
        return frame
    end_epoch = int(start_epoch) + int(max_epochs)
    in_window = (frame[epoch_column] > int(start_epoch)) & (frame[epoch_column] <= end_epoch)
    if next_epoch_column is not None:
        has_next = frame[next_epoch_column] > 0
        next_in_window = (~has_next) | (frame[next_epoch_column] <= end_epoch)
        in_window &= next_in_window
    return frame[in_window].copy()
