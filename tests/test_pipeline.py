import json

import numpy as np
import pytest

from twb.audio import read_audio
from twb.cli import main
from twb.conditions import Condition, apply_condition, condition_from_dict, derive_seed, load_conditions
from twb.datasets import bundled_sample_dir, load_source
from twb.prepare import prepare, read_manifest
from twb.report import build_report, load_run, render_markdown, score_row, write_report
from twb.runner import run_manifest

CONDS = [
    {"name": "clean-16k", "sample_rate": 16000},
    {"name": "mulaw-8k", "sample_rate": 8000, "codec": "mulaw"},
    {"name": "mulaw-8k-loss10", "sample_rate": 8000, "codec": "mulaw", "loss": 0.1},
    {"name": "snr10", "sample_rate": 8000, "codec": "mulaw", "noise": {"type": "pink", "snr_db": 10}},
]


def test_bundled_sample_loads_and_has_license():
    items = load_source(None)
    assert len(items) == 10
    assert all(i.audio.exists() and i.text for i in items)
    assert (bundled_sample_dir() / "LICENSE.txt").read_text().startswith("Audio and transcripts")


def test_condition_validation():
    with pytest.raises(ValueError):
        condition_from_dict({"name": "bad", "sample_rate": 16000, "codec": "mulaw"})
    with pytest.raises(ValueError):
        condition_from_dict({"name": "bad", "sample_rate": 8000, "noise": {"type": "pink"}})
    with pytest.raises(ValueError):
        condition_from_dict({"name": "bad", "sample_rate": 11025})


def test_example_conditions_files_parse():
    from pathlib import Path

    root = Path(__file__).parent.parent / "examples"
    assert len(load_conditions(root / "conditions.yaml")) == 4
    assert len(load_conditions(root / "conditions-extended.yaml")) >= 8


def test_derive_seed_is_stable():
    assert derive_seed(1, "a", "b") == derive_seed(1, "a", "b")
    assert derive_seed(1, "a", "b") != derive_seed(2, "a", "b")


def test_apply_condition_rates_and_determinism(speechish):
    audio, sr = speechish
    c = Condition(name="x", sample_rate=8000, codec="mulaw", loss=0.2)
    a = apply_condition(c, audio, sr, seed=5)
    b = apply_condition(c, audio, sr, seed=5)
    d = apply_condition(c, audio, sr, seed=6)
    assert a.sample_rate == 8000 and len(a.audio) == len(audio) // 2
    assert np.array_equal(a.audio, b.audio) and not np.array_equal(a.audio, d.audio)


def test_prepare_writes_manifest_and_is_reproducible(tmp_path):
    items = load_source(None)[:3]
    conds = [condition_from_dict(c) for c in CONDS]
    m1 = prepare(items, conds, tmp_path / "a", seed=11)
    prepare(items, conds, tmp_path / "b", seed=11)
    rows = read_manifest(m1)
    assert len(rows) == 3 * 4
    r = rows[0]
    for key in ("id", "utterance", "condition", "audio", "text", "sample_rate", "speech_start_s",
                "speech_end_s", "seed", "params"):
        assert key in r
    # same seed, same audio, byte for byte
    for row in rows:
        assert (tmp_path / "a" / row["audio"]).read_bytes() == (tmp_path / "b" / row["audio"]).read_bytes()
    assert (tmp_path / "a" / "prepare.json").exists()
    # 8 kHz files really are 8 kHz and the same duration as the source
    audio, sr = read_audio(tmp_path / "a" / rows[1]["audio"])
    assert sr == 8000 and rows[1]["condition"] == "mulaw-8k"
    assert abs(len(audio) / sr - rows[0]["duration_s"]) < 0.03


def test_prepare_different_seed_changes_lossy_audio(tmp_path):
    items = load_source(None)[:1]
    conds = [condition_from_dict(CONDS[2])]
    a = prepare(items, conds, tmp_path / "a", seed=1)
    b = prepare(items, conds, tmp_path / "b", seed=2)
    ra, rb = read_manifest(a)[0], read_manifest(b)[0]
    assert (tmp_path / "a" / ra["audio"]).read_bytes() != (tmp_path / "b" / rb["audio"]).read_bytes()


@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("oracle")
    items = load_source(None)[:3]
    conds = [condition_from_dict(c) for c in CONDS[:2]]
    prepare(items, conds, root / "prep", seed=3)
    run_manifest(root / "prep", {"name": "oracle"}, root / "run", workers=1, tail_s=0.1, pace=False)
    return root


def test_run_writes_results(oracle_run):
    meta, rows = load_run(oracle_run / "run.json" if False else oracle_run / "run")
    assert meta["errors"] == 0 and len(rows) == 6
    assert rows[0]["events"][-1]["kind"] == "final"
    assert meta["provider"]["name"] == "oracle"


def test_oracle_scores_zero_wer(oracle_run):
    _, rows = load_run(oracle_run / "run")
    for r in rows:
        s = score_row(r)
        assert s["wer_counts"].errors == 0


def test_report_files_and_content(oracle_run, tmp_path):
    report = build_report([oracle_run / "run"])
    assert report["baseline_condition"] == "clean-16k"
    cond = report["providers"]["oracle"]["conditions"]["mulaw-8k"]
    assert cond["wer"] == 0.0 and cond["utterances"] == 3
    md = render_markdown(report)
    assert "| mulaw-8k |" in md and "oracle" in md
    paths = write_report(report, tmp_path / "rep")
    names = {p.name for p in paths}
    assert names == {"report.json", "report.md", "wer_by_condition.png", "latency_by_condition.png"}
    assert all(p.stat().st_size > 0 for p in paths)
    json.loads((tmp_path / "rep" / "report.json").read_text())


def test_cli_end_to_end(tmp_path, capsys):
    cond = tmp_path / "c.yaml"
    cond.write_text("conditions:\n  - {name: clean-16k, sample_rate: 16000}\n"
                    "  - {name: mulaw-8k, sample_rate: 8000, codec: mulaw}\n")
    assert main(["prepare", "--conditions", str(cond), "--out", str(tmp_path / "p"), "--limit", "2"]) == 0
    assert main(["run", str(tmp_path / "p"), "--provider", "oracle", "--no-pace", "--tail", "0.1",
                 "--out", str(tmp_path / "r")]) == 0
    assert main(["report", str(tmp_path / "r"), "--out", str(tmp_path / "rep"), "--no-charts"]) == 0
    out = capsys.readouterr().out
    assert "mulaw-8k" in out and "WER" in out
    assert (tmp_path / "rep" / "report.json").exists()
