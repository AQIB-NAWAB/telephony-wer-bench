"""Plain matplotlib charts. White background, thin axes, one muted palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#1f5fa8", "#8a8a8a", "#b5651d", "#3a7d44", "#7a4a8a"]


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


def _sample_note(report: dict) -> str:
    parts = []
    for name, p in report["providers"].items():
        parts.append(f"{name}: {p['run'].get('utterances', '?')} utterances per condition")
    return "; ".join(parts)


def plot_wer(report: dict, path: Path) -> Path:
    conds = report["condition_order"]
    provs = list(report["providers"].items())
    wide = len(conds) > 6
    fig, ax = plt.subplots(figsize=(14 if wide else 10, 4.8), dpi=130)
    width = 0.8 / max(1, len(provs))
    for k, (name, p) in enumerate(provs):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            d = p["conditions"].get(c)
            if not d:
                continue
            xs.append(i + (k - (len(provs) - 1) / 2) * width)
            ys.append(d["wer"] * 100)
            ci = d.get("wer_ci95")
            lo.append(d["wer"] * 100 - (ci[0] * 100 if ci else d["wer"] * 100))
            hi.append((ci[1] * 100 if ci else d["wer"] * 100) - d["wer"] * 100)
        ax.bar(xs, ys, width * 0.92, color=PALETTE[k % len(PALETTE)], label=name,
               yerr=[lo, hi], error_kw={"ecolor": "#333333", "elinewidth": 1, "capsize": 3})
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, fontsize=8 if wide else 9, rotation=20 if wide else 0)
    ax.set_ylabel("Word error rate (%)")
    ax.set_title("WER by condition (streaming, final transcript)", loc="left", fontsize=12)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, f"{_sample_note(report)}. Bars show 95% bootstrap CI over utterances.",
             fontsize=8, color="#666666")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_latency(report: dict, path: Path) -> Path:
    conds = report["condition_order"]
    provs = list(report["providers"].items())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
    specs = [("time_to_first_partial", "Time to first partial (ms, p50)"),
             ("final_latency_after_speech_end", "Final transcript after end of speech (ms, p50)")]
    width = 0.8 / max(1, len(provs))
    for ax, (key, title) in zip(axes, specs):
        for k, (name, p) in enumerate(provs):
            xs, ys = [], []
            for i, c in enumerate(conds):
                d = p["conditions"].get(c)
                if d and d.get(key):
                    xs.append(i + (k - (len(provs) - 1) / 2) * width)
                    ys.append(d[key]["p50_ms"])
            ax.bar(xs, ys, width * 0.92, color=PALETTE[k % len(PALETTE)], label=name)
            for x, y in zip(xs, ys):
                ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=8, rotation=15)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=len(labels))
    fig.text(0.01, 0.01,
             f"{_sample_note(report)}. Real-time paced, 20 ms frames. Latency depends on this machine.",
             fontsize=8, color="#666666")
    fig.subplots_adjust(bottom=0.2, top=0.86)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
