"""Regenerate the images in docs/images from the committed results.

    pip install -e ".[dev]"
    python scripts/make_images.py

Nothing is typed in by hand. The terminal image is the real output of `twb report` on the
committed run folders in results/. The charts are drawn from the same runs. The spectrogram
image is made by running `twb prepare` on the bundled sample, so it needs no download.
The architecture diagram is drawn from a fixed layout.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from twb.audio import read_audio  # noqa: E402
from twb.conditions import condition_from_dict  # noqa: E402
from twb.datasets import load_source  # noqa: E402
from twb.plots import plot_latency, plot_wer  # noqa: E402
from twb.prepare import prepare, read_manifest  # noqa: E402
from twb.report import build_report, write_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
RESULTS = ROOT / "results"
HEADLINE = ["clean-16k", "mulaw-8k", "mulaw-8k-loss5", "mulaw-8k-snr10"]
RUNS = ["vosk", "whisper-base-en"]
INK, GRAY, ACCENT, SOFT = "#1a1a1a", "#6b6b6b", "#1f5fa8", "#c9d9ec"


def save_png(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(colors=64).save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def terminal_png(name: str, command: str, output: str) -> None:
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), 16)
    lines = [("$ " + command, True)] + [(line, False) for line in output.splitlines()]
    pad, lh = 22, 24
    width = max(int(font.getlength(t)) for t, _ in lines) + pad * 2
    img = Image.new("RGB", (max(width, 760), lh * len(lines) + pad * 2), "#1c1c1c")
    d = ImageDraw.Draw(img)
    for i, (text, is_cmd) in enumerate(lines):
        y = pad + i * lh
        if is_cmd:
            d.text((pad, y), "$ ", font=font, fill="#8fbf8f")
            d.text((pad + font.getlength("$ "), y), text[2:], font=font, fill="#f2f2f2")
        else:
            d.text((pad, y), text, font=font, fill="#cfcfcf")
    path = OUT / name
    img.quantize(colors=32).save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def spectrograms() -> None:
    cond_defs = [
        {"name": "clean-16k", "sample_rate": 16000},
        {"name": "mulaw-8k", "sample_rate": 8000, "codec": "mulaw"},
        {"name": "mulaw-8k-loss10", "sample_rate": 8000, "codec": "mulaw", "loss": 0.10},
        {"name": "mulaw-8k-snr10", "sample_rate": 8000, "codec": "mulaw", "noise": {"type": "pink", "snr_db": 10}},
    ]
    items = load_source(None)[:1]
    with tempfile.TemporaryDirectory() as tmp:
        prepare(items, [condition_from_dict(c) for c in cond_defs], tmp, seed=1234)
        rows = read_manifest(tmp)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), dpi=110, sharex=True, sharey=True)
        for ax, row in zip(axes.flat, rows):
            audio, sr = read_audio(Path(tmp) / row["audio"])
            ax.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="Greys", vmin=-120, vmax=-30)
            ax.set_title(row["condition"], loc="left", fontsize=11)
            ax.set_ylim(0, 8000)
            if row["sample_rate"] == 8000:
                ax.axhline(4000, color=ACCENT, linewidth=1)
        for ax in axes[:, 0]:
            ax.set_ylabel("Hz")
        for ax in axes[1, :]:
            ax.set_xlabel("seconds")
        fig.text(0.01, 0.01, "Same utterance (LibriSpeech 1221-135766-0013), four conditions. Blue line: 4 kHz, "
                 "the top of an 8 kHz signal.", fontsize=9, color=GRAY)
        fig.subplots_adjust(bottom=0.12, hspace=0.25)
        save_png(fig, "spectrograms.png")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=110)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=GRAY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    linewidth=1.2, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color=GRAY))

    box(1, 28, 20, 9, "source audio\n+ transcripts")
    box(1, 8, 20, 9, "conditions.yaml\n(codec, loss, noise)")
    box(29, 17, 18, 11, "twb prepare\nresample, G.711,\nloss, noise", fill=SOFT, edge=ACCENT)
    box(55, 17, 16, 11, "manifest.jsonl\n+ WAV files")
    box(79, 29, 29, 9, "twb run\n20 ms frames, real time", fill=SOFT, edge=ACCENT)
    box(79, 6, 29, 14, "STT provider\nVosk, faster-whisper\n(cloud adapters untested)")
    box(55, 1, 16, 8, "results.jsonl\npartials, finals, times")
    box(29, 1, 18, 8, "twb report\nWER, CER, latency", fill=SOFT, edge=ACCENT)
    arrow(21, 32, 29, 25)
    arrow(21, 13, 29, 20)
    arrow(47, 22, 55, 22)
    arrow(71, 25, 79, 32)
    arrow(93, 29, 93, 20)
    arrow(79, 10, 71, 5)
    arrow(55, 5, 47, 5)
    ax.text(100, 24.5, "audio frames", fontsize=8, color=GRAY, ha="center")
    ax.text(100, 22.0, "partial and final", fontsize=8, color=GRAY, ha="center")
    ax.text(100, 20.5, "events back", fontsize=8, color=GRAY, ha="center")
    save_png(fig, "architecture.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [RESULTS / r for r in RUNS if (RESULTS / r / "results.jsonl").exists()]
    if not runs:
        print("no runs in results/. Run the benchmark first, see README.", file=sys.stderr)
        return 1
    report = build_report(runs, only=HEADLINE)
    write_report(report, RESULTS / "report", charts=False)
    plot_wer(report, OUT / "wer-by-condition.png")
    plot_latency(report, OUT / "latency-by-condition.png")
    for name in ("wer-by-condition.png", "latency-by-condition.png"):
        p = OUT / name
        Image.open(p).convert("RGB").quantize(colors=64).save(p, optimize=True)
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
    full = build_report(runs)
    write_report(full, RESULTS / "report-all", charts=False)
    plot_wer(full, OUT / "wer-all-conditions.png")
    p = OUT / "wer-all-conditions.png"
    Image.open(p).convert("RGB").quantize(colors=64).save(p, optimize=True)
    print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
    cmd = ["twb", "report", *[str(r.relative_to(ROOT)) for r in runs], "--only", *HEADLINE, "--no-charts",
           "--out", "/tmp/twb-terminal-report"]
    proc = subprocess.run([sys.executable, "-m", "twb.cli", *cmd[1:]], cwd=ROOT, capture_output=True, text=True,
                          check=True)
    shown = proc.stdout.split("\nwrote:")[0].rstrip()
    terminal_png("terminal-report.png", "twb report results/vosk results/whisper-base-en --only " + " ".join(HEADLINE),
                 shown)
    spectrograms()
    architecture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
