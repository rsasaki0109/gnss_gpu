"""Generate the GitHub social-preview card (Open Graph image).

Produces a 1280x640 PNG used as the repo's "Social preview" image (the thumbnail
shown when the repo link is shared on X / Slack / Hacker News / LinkedIn, etc.).

    python3 experiments/build_social_preview.py

Output: docs/assets/media/social_preview.png

Upload is a one-time manual step (GitHub has no API for it):
    GitHub repo -> Settings -> General -> Social preview -> Upload an image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "assets" / "media" / "social_preview.png"

# GitHub renders social previews at 1280x640 (2:1).
WIDTH_PX, HEIGHT_PX = 1280, 640
DPI = 100

BG = "#0b1021"
ACCENT = "#36d399"
TEXT = "#e8ecf5"
MUTED = "#9aa6c0"
CHIP_BG = "#161d36"


def _chip(ax, x, y, w, h, value, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0, facecolor=CHIP_BG, transform=ax.transAxes,
        )
    )
    ax.text(x + w / 2, y + h * 0.62, value, transform=ax.transAxes,
            ha="center", va="center", color=ACCENT, fontsize=21, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.24, label, transform=ax.transAxes,
            ha="center", va="center", color=MUTED, fontsize=11.5)


def main() -> None:
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Accent bar
    ax.add_patch(FancyBboxPatch((0.06, 0.88), 0.10, 0.012, boxstyle="square,pad=0",
                                linewidth=0, facecolor=ACCENT, transform=ax.transAxes))

    ax.text(0.06, 0.79, "gnss_gpu", transform=ax.transAxes, color=TEXT,
            fontsize=52, fontweight="bold", family="monospace")

    ax.text(0.06, 0.66, "Beat RTKLIB in the urban canyon", transform=ax.transAxes,
            color=ACCENT, fontsize=30, fontweight="bold")
    ax.text(0.06, 0.575, "GPU particle-filter GNSS positioning with ray-traced NLOS rejection",
            transform=ax.transAxes, color=MUTED, fontsize=16.5)

    # Metric chips
    y, h, w, gap = 0.20, 0.24, 0.265, 0.035
    x0 = 0.06
    _chip(ax, x0, y, w, h, "1.36 / 4.11 m", "PF P50 / RMS  (UrbanNav Odaiba)")
    _chip(ax, x0 + (w + gap), y, w, h, "vs 2.67 / 13.08 m", "RTKLIB demo5  (same epochs)")
    _chip(ax, x0 + 2 * (w + gap), y, w, h, "81 ms", "1,000,000-particle filter step")

    ax.text(0.06, 0.08, "github.com/rsasaki0109/gnss_gpu", transform=ax.transAxes,
            color=MUTED, fontsize=14, family="monospace")
    ax.text(0.94, 0.08, "CUDA  +  Python  •  Apache-2.0", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {OUT_PATH} ({WIDTH_PX}x{HEIGHT_PX})")


if __name__ == "__main__":
    main()
