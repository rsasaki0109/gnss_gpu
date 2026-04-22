#!/usr/bin/env python3
"""Build lightweight poster and teaser media for the repo front door."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ASSETS_DIR = PROJECT_ROOT / "experiments" / "results" / "paper_assets"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MEDIA_DIR = PROJECT_ROOT / "docs" / "assets" / "media"

POSTER_PATH = MEDIA_DIR / "site_poster.png"
TEASER_PATH = MEDIA_DIR / "site_teaser.gif"
TEASER_MP4_PATH = MEDIA_DIR / "site_teaser.mp4"
TEASER_WEBM_PATH = MEDIA_DIR / "site_teaser.webm"
URBANNAV_RUNS_CHART_PATH = MEDIA_DIR / "site_urbannav_runs.png"
WINDOW_WINS_CHART_PATH = MEDIA_DIR / "site_window_wins.png"
HK_CONTROL_CHART_PATH = MEDIA_DIR / "site_hk_control.png"
URBANNAV_TIMELINE_CHART_PATH = MEDIA_DIR / "site_urbannav_timeline.png"
ERROR_BANDS_CHART_PATH = MEDIA_DIR / "site_error_bands.png"

URBANNAV_RUNS_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv"
WINDOW_SUMMARY_CSV = "urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv"
HK_CONTROL_CSV = "urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv"
URBANNAV_EPOCHS_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_epochs_epochs.csv"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _rounded_panel(base: Image.Image, box: tuple[int, int, int, int], fill: str) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box, radius=34, fill=fill)
    base.alpha_composite(overlay)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[tuple[str, ImageFont.ImageFont, str]],
    spacing: int = 10,
) -> None:
    x, y = xy
    for text, font, fill in lines:
        if "\n" in text:
            draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6)
            bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=6)
        else:
            draw.text((x, y), text, font=font, fill=fill)
            bbox = draw.textbbox((x, y), text, font=font)
        y = bbox[3] + spacing


def _load_and_fit(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    acc = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        acc += value
        if len(queue) > window:
            acc -= queue.pop(0)
        out.append(acc / len(queue))
    return out


def _build_poster() -> Image.Image:
    width, height = 1600, 900
    bg = Image.new("RGBA", (width, height), "#ece5d7")
    for y in range(height):
        blend = y / max(height - 1, 1)
        r = int(236 * (1 - blend) + 214 * blend)
        g = int(229 * (1 - blend) + 222 * blend)
        b = int(215 * (1 - blend) + 210 * blend)
        ImageDraw.Draw(bg).line([(0, y), (width, y)], fill=(r, g, b, 255))

    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((1020, -80, 1560, 460), fill=(12, 123, 116, 52))
    glow_draw.ellipse((1020, 430, 1580, 940), fill=(188, 108, 37, 42))
    glow = glow.filter(ImageFilter.GaussianBlur(36))
    bg.alpha_composite(glow)

    poster = bg.copy()
    _rounded_panel(poster, (48, 42, 716, 358), "#fffaf1")
    _rounded_panel(poster, (744, 42, 1552, 248), "#168f86")
    _rounded_panel(poster, (744, 250, 1130, 852), "#fffaf1")
    _rounded_panel(poster, (1158, 250, 1552, 552), "#fffaf1")
    _rounded_panel(poster, (1158, 580, 1552, 852), "#fffaf1")

    draw = ImageDraw.Draw(poster)
    ink = "#1f2724"
    muted = "#55615d"
    teal = "#0f766e"
    rust = "#bc6c25"

    _draw_text_block(
        draw,
        (92, 82),
        [
            ("GNSS_GPU", _font(30, bold=True), rust),
            ("gnss_gpu\nArtifact\nSnapshot", _font(52, bold=True), ink),
            (
                "Frozen mainline: PF+RobustClear-10K",
                _font(23),
                muted,
            ),
        ],
        spacing=12,
    )
    _draw_text_block(
        draw,
        (788, 78),
        [
            ("CURRENT READ", _font(22, bold=True), "#d6fffa"),
            ("66.60 m external RMS\n57.8x BVH speedup", _font(40, bold=True), "#ffffff"),
            (
                "UrbanNav trimble + G,E,J is the frozen headline.\nPPC holdout stays smaller, while BVH carries the systems story.",
                _font(17),
                "#e7fffc",
            ),
        ],
        spacing=10,
    )

    figure_specs = [
        (
            PAPER_ASSETS_DIR / "paper_urbannav_external.png",
            (774, 280, 1100, 568),
            "UrbanNav External",
            "Main accuracy figure",
        ),
        (
            PAPER_ASSETS_DIR / "paper_bvh_runtime.png",
            (1188, 280, 1522, 458),
            "BVH Runtime",
            "Same accuracy, 57.8x faster",
        ),
        (
            PAPER_ASSETS_DIR / "paper_ppc_holdout.png",
            (1188, 610, 1522, 792),
            "PPC Holdout",
            "Exploratory gate survives holdout",
        ),
    ]

    for path, box, title, subtitle in figure_specs:
        x0, y0, x1, y1 = box
        pad = 18
        img = _load_and_fit(path, (x1 - x0 - pad * 2, y1 - y0 - pad * 2 - 54))
        poster.alpha_composite(img, (x0 + pad, y0 + pad))
        draw.rounded_rectangle((x0 + 16, y1 - 52, x1 - 16, y1 - 16), radius=18, fill="#fff4e6")
        draw.text((x0 + 28, y1 - 48), title, font=_font(24, bold=True), fill=ink)
        draw.text((x0 + 28, y1 - 22), subtitle, font=_font(18), fill=muted)

    metric_boxes = [
        ((90, 288, 330, 356), "PF+RobustClear", "10K frozen winner"),
        ((350, 288, 560, 356), "93.25 -> 66.60", "UrbanNav vs EKF"),
        ((580, 288, 692, 356), "440/7", "tests/skips"),
    ]
    for box, value, label in metric_boxes:
        draw.rounded_rectangle(box, radius=20, fill=ImageColor.getrgb("#efe5d4"))
        draw.text((box[0] + 16, box[1] + 10), value, font=_font(21, bold=True), fill=ink)
        draw.text((box[0] + 18, box[1] + 38), label, font=_font(16), fill=muted)

    return poster


def _build_teaser(poster: Image.Image) -> list[Image.Image]:
    target_size = (1200, 675)
    base = poster.resize(target_size, Image.Resampling.LANCZOS).convert("RGBA")

    focus_specs = [
        {
            "image_path": PAPER_ASSETS_DIR / "paper_urbannav_external.png",
            "eyebrow": "MAINLINE",
            "title": "UrbanNav external win",
            "subtitle": "PF+RobustClear-10K beats EKF on trimble + G,E,J.",
            "metrics": [("66.60 m", "RMS 2D"), ("98.53 m", "P95")],
            "footer": "External Tokyo result, frozen mainline headline.",
            "accent": "#168f86",
            "panel": "#fffaf1",
            "chip": "#dff6f3",
        },
        {
            "image_path": PAPER_ASSETS_DIR / "paper_bvh_runtime.png",
            "eyebrow": "SYSTEMS",
            "title": "BVH systems speedup",
            "subtitle": "Same PF3D path, 57.8x faster than the linear baseline.",
            "metrics": [("1028.29", "ms/epoch"), ("17.78", "BVH ms/epoch")],
            "footer": "Acceleration is a main contribution, not a side note.",
            "accent": "#bc6c25",
            "panel": "#fff7ef",
            "chip": "#fce6d0",
        },
        {
            "image_path": PAPER_ASSETS_DIR / "paper_ppc_holdout.png",
            "eyebrow": "HOLDOUT",
            "title": "PPC holdout signal",
            "subtitle": "Exploratory gate survives holdout, but stays supplemental.",
            "metrics": [("66.92", "baseline RMS"), ("65.54", "best holdout RMS")],
            "footer": "Design-space evidence, not the external headline.",
            "accent": "#3b5b75",
            "panel": "#f6f8fb",
            "chip": "#dbe8f1",
        },
    ]

    frames = [base.convert("P", palette=Image.Palette.ADAPTIVE)]
    for index, spec in enumerate(focus_specs, start=1):
        frame = Image.new("RGBA", target_size, "#efe7d8")

        for y in range(target_size[1]):
            blend = y / max(target_size[1] - 1, 1)
            r = int(239 * (1 - blend) + 227 * blend)
            g = int(231 * (1 - blend) + 235 * blend)
            b = int(216 * (1 - blend) + 223 * blend)
            ImageDraw.Draw(frame).line([(0, y), (target_size[0], y)], fill=(r, g, b, 255))

        glow = Image.new("RGBA", target_size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        accent = ImageColor.getrgb(spec["accent"])
        glow_draw.ellipse((700, -40, 1280, 470), fill=accent + (58,))
        glow_draw.ellipse((-120, 430, 360, 860), fill=(255, 250, 241, 84))
        glow = glow.filter(ImageFilter.GaussianBlur(40))
        frame.alpha_composite(glow)

        frame_draw = ImageDraw.Draw(frame)
        shadow = Image.new("RGBA", target_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        preview_box = (516, 74, 1134, 456)
        shadow_draw.rounded_rectangle(
            (preview_box[0] + 12, preview_box[1] + 18, preview_box[2] + 12, preview_box[3] + 18),
            radius=42,
            fill=(36, 41, 39, 44),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        frame.alpha_composite(shadow)

        frame_draw.rounded_rectangle(preview_box, radius=38, fill=spec["panel"])
        frame_draw.rounded_rectangle((548, 102, 654, 138), radius=16, fill=spec["chip"])
        frame_draw.text((572, 110), f"0{index}", font=_font(20, bold=True), fill=spec["accent"])

        preview = _load_and_fit(spec["image_path"], (preview_box[2] - preview_box[0] - 42, 250))
        frame.alpha_composite(preview, (preview_box[0] + 21, preview_box[1] + 56))

        frame_draw.rounded_rectangle(
            (preview_box[0] + 22, preview_box[1] + 322, preview_box[2] - 22, preview_box[3] - 20),
            radius=24,
            fill="#fff4e6",
        )
        frame_draw.text(
            (preview_box[0] + 44, preview_box[1] + 338),
            spec["title"],
            font=_font(30, bold=True),
            fill="#1f2724",
        )
        frame_draw.text(
            (preview_box[0] + 44, preview_box[1] + 378),
            spec["footer"],
            font=_font(18),
            fill="#55615d",
        )

        frame_draw.rounded_rectangle((70, 74, 458, 504), radius=38, fill=(255, 250, 241, 224))
        frame_draw.rounded_rectangle((94, 98, 248, 140), radius=18, fill=spec["accent"])
        frame_draw.text((118, 108), spec["eyebrow"], font=_font(20, bold=True), fill="#f2fffd")
        _draw_text_block(
            frame_draw,
            (96, 172),
            [
                (spec["title"], _font(44, bold=True), "#1f2724"),
                (spec["subtitle"], _font(24), "#55615d"),
            ],
            spacing=14,
        )

        chip_y = 360
        for metric_value, metric_label in spec["metrics"]:
            chip_box = (96, chip_y, 412, chip_y + 74)
            frame_draw.rounded_rectangle(chip_box, radius=24, fill=spec["chip"])
            frame_draw.text((118, chip_y + 12), metric_value, font=_font(28, bold=True), fill="#1f2724")
            frame_draw.text((118, chip_y + 44), metric_label, font=_font(17), fill="#55615d")
            chip_y += 88

        frame_draw.rounded_rectangle((70, 548, 1130, 624), radius=26, fill=(255, 250, 241, 226))
        frame_draw.text(
            (98, 570),
            "artifact snapshot: frozen headline, systems result, and holdout context",
            font=_font(24, bold=True),
            fill="#273330",
        )
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
    return frames


def _build_teaser_video() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(TEASER_PATH),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=1200:-2:flags=lanczos",
            str(TEASER_MP4_PATH),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(TEASER_PATH),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-vf",
            "scale=1200:-2:flags=lanczos",
            str(TEASER_WEBM_PATH),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _plot_urbannav_runs() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(URBANNAV_RUNS_CSV)
    runs = ["Odaiba", "Shinjuku"]
    methods = ["EKF", "PF-10K", "PF+RobustClear-10K"]
    colors = {
        "EKF": "#3b82f6",
        "PF-10K": "#f97316",
        "PF+RobustClear-10K": "#059669",
    }

    metrics = [
        ("rms_2d", "RMS 2D [m]"),
        ("p95", "P95 [m]"),
    ]
    x = np.arange(len(runs))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for idx, method in enumerate(methods):
            vals = [
                float(next(row[metric] for row in rows if row["run"] == run and row["method"] == method))
                for run in runs
            ]
            ax.bar(x + (idx - 1) * width, vals, width=width, color=colors[method], label=method)
        ax.set_xticks(x, runs)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_title("UrbanNav per-run RMS")
    axes[1].set_title("UrbanNav per-run P95")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("UrbanNav Tokyo: per-run comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(URBANNAV_RUNS_CHART_PATH, dpi=180)
    plt.close(fig)


def _plot_window_wins() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(WINDOW_SUMMARY_CSV)
    methods = ["PF-10K", "PF+RobustClear-10K"]
    labels = ["RMS", "P95", ">100 m", ">500 m"]
    keys = ["win_rate_rms_pct", "win_rate_p95_pct", "win_rate_outlier_pct", "win_rate_catastrophic_pct"]
    colors = ["#f97316", "#059669"]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for idx, method in enumerate(methods):
        row = next(item for item in rows if item["method"] == method)
        vals = [float(row[key]) for key in keys]
        ax.barh(y + (idx - 0.5) * 0.28, vals, height=0.26, color=colors[idx], label=method)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Win rate vs EKF [%]")
    ax.set_title("UrbanNav fixed-window wins vs EKF")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(WINDOW_WINS_CHART_PATH, dpi=180)
    plt.close(fig)


def _plot_hk_control() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(HK_CONTROL_CSV)
    methods = ["EKF", "WLS+QualityVeto", "PF-10K", "PF+AdaptiveGuide-10K"]
    colors = ["#3b82f6", "#8b5cf6", "#f97316", "#059669"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title in [
        (axes[0], "mean_rms_2d", "Hong Kong control RMS"),
        (axes[1], "mean_p95", "Hong Kong control P95"),
    ]:
        vals = [float(next(row[metric] for row in rows if row["method"] == method)) for method in methods]
        x = np.arange(len(methods))
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x, ["EKF", "WLS+QV", "PF-10K", "PF+Adaptive"])
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Hong Kong control: mitigation helps, but stays supplemental", fontsize=13)
    fig.tight_layout()
    fig.savefig(HK_CONTROL_CHART_PATH, dpi=180)
    plt.close(fig)


def _plot_urbannav_timeline() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(URBANNAV_EPOCHS_CSV)
    methods = ["EKF", "PF+RobustClear-10K"]
    colors = {"EKF": "#3b82f6", "PF+RobustClear-10K": "#059669"}
    runs = ["Odaiba", "Shinjuku"]

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.8), sharex=False)
    for ax, run in zip(axes, runs, strict=True):
        for method in methods:
            method_rows = [
                row for row in rows if row["run"] == run and row["method"] == method
            ]
            epoch = [int(row["epoch_index"]) for row in method_rows]
            error = [float(row["error_2d"]) for row in method_rows]
            smooth = _rolling_mean(error, window=180)
            ax.plot(epoch, smooth, color=colors[method], linewidth=2.1, label=method)
        ax.axhline(100.0, color="#bc6c25", linewidth=1.1, linestyle="--", alpha=0.8)
        ax.text(8, 104.0, "100 m", color="#8f4e16", fontsize=9)
        ax.set_title(f"{run}: smoothed epoch error")
        ax.set_ylabel("2D error [m]")
        ax.grid(True, alpha=0.22)
        ax.set_ylim(0, 210)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Epoch index")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("UrbanNav Tokyo: epoch-level error trace", fontsize=14)
    fig.tight_layout()
    fig.savefig(URBANNAV_TIMELINE_CHART_PATH, dpi=180)
    plt.close(fig)


def _plot_error_bands() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(URBANNAV_EPOCHS_CSV)
    methods = ["EKF", "PF-10K", "PF+RobustClear-10K"]
    method_colors = {
        "<25 m": "#d8f3dc",
        "25-50 m": "#95d5b2",
        "50-100 m": "#52b788",
        "100-500 m": "#f4a261",
        ">500 m": "#e63946",
    }
    bands = [
        ("<25 m", lambda e: e < 25.0),
        ("25-50 m", lambda e: 25.0 <= e < 50.0),
        ("50-100 m", lambda e: 50.0 <= e < 100.0),
        ("100-500 m", lambda e: 100.0 <= e < 500.0),
        (">500 m", lambda e: e >= 500.0),
    ]

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for label, pred in bands:
        vals = []
        for method in methods:
            errors = [
                float(row["error_2d"])
                for row in rows
                if row["method"] == method and row["run"] in {"Odaiba", "Shinjuku"}
            ]
            pct = 100.0 * sum(1 for e in errors if pred(e)) / max(len(errors), 1)
            vals.append(pct)
        ax.barh(y, vals, left=left, color=method_colors[label], label=label)
        left += np.array(vals)

    ax.set_yticks(y, methods)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Epoch share [%]")
    ax.set_title("UrbanNav external: error-band composition")
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(ERROR_BANDS_CHART_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    poster = _build_poster()
    poster.save(POSTER_PATH, optimize=True)

    teaser_frames = _build_teaser(poster)
    teaser_frames[0].save(
        TEASER_PATH,
        save_all=True,
        append_images=teaser_frames[1:],
        duration=[1200, 1100, 1100, 1100],
        loop=0,
        optimize=True,
        disposal=2,
    )
    _build_teaser_video()
    _plot_urbannav_runs()
    _plot_window_wins()
    _plot_hk_control()
    _plot_urbannav_timeline()
    _plot_error_bands()
    print(f"wrote {POSTER_PATH}")
    print(f"wrote {TEASER_PATH}")
    if TEASER_MP4_PATH.exists():
        print(f"wrote {TEASER_MP4_PATH}")
    if TEASER_WEBM_PATH.exists():
        print(f"wrote {TEASER_WEBM_PATH}")
    print(f"wrote {URBANNAV_RUNS_CHART_PATH}")
    print(f"wrote {WINDOW_WINS_CHART_PATH}")
    print(f"wrote {HK_CONTROL_CHART_PATH}")
    print(f"wrote {URBANNAV_TIMELINE_CHART_PATH}")
    print(f"wrote {ERROR_BANDS_CHART_PATH}")


if __name__ == "__main__":
    main()
