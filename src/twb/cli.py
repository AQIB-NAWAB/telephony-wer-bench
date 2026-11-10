"""Command line entry point: ``twb``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _parse_opts(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--opt expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    from .datasets import LIBRISPEECH_CITATION, fetch_librispeech

    items = fetch_librispeech(args.out, n=args.n, seed=args.seed, min_s=args.min_s, max_s=args.max_s,
                              tarball=args.tarball)
    print(f"wrote {len(items)} utterances to {args.out}")
    print("Source: LibriSpeech test-clean, CC BY 4.0.")
    print(LIBRISPEECH_CITATION)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .conditions import load_conditions
    from .datasets import bundled_sample_dir, load_source
    from .prepare import prepare

    cond_path = args.conditions
    if cond_path is None:
        raise SystemExit("pass --conditions path/to/conditions.yaml (see examples/conditions.yaml)")
    items = load_source(args.source)
    if args.limit:
        items = items[: args.limit]
    conds = load_conditions(cond_path)
    src_name = str(args.source or bundled_sample_dir())
    manifest = prepare(items, conds, args.out, seed=args.seed, source_name=src_name)
    print(f"prepared {len(items)} utterances x {len(conds)} conditions -> {manifest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_manifest

    spec = {"name": args.provider, "options": _parse_opts(args.opt)}

    def progress(k: int, n: int) -> None:
        if k % 10 == 0 or k == n:
            print(f"  {k}/{n}", file=sys.stderr, flush=True)

    out = run_manifest(args.manifest, spec, args.out, workers=args.workers, tail_s=args.tail,
                       pace=not args.no_pace, limit=args.limit, only=args.only, progress=progress)
    meta = json.loads((Path(out) / "run.json").read_text())
    print(f"run finished in {meta['wall_seconds']}s, {meta['errors']} errors -> {out}")
    if args.no_pace:
        print("warning: --no-pace was used, so latency numbers are not meaningful", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, render_text, write_report

    report = build_report(args.runs, baseline=args.baseline, only=args.only)
    paths = write_report(report, args.out, charts=not args.no_charts)
    print(render_text(report))
    print("\nwrote:")
    for p in paths:
        print(f"  {p}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    from .providers import PROVIDERS, make_provider

    for name in PROVIDERS:
        try:
            p = make_provider(name)
            status = "ready" if p.tested else "ready (adapter untested)"
        except Exception as exc:
            status = f"not available: {exc}"
        print(f"{name:16s} {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twb", description="Telephony WER bench for streaming speech-to-text.")
    p.add_argument("--version", action="version", version=f"twb {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download LibriSpeech test-clean and keep a small sample")
    f.add_argument("--out", default="twb-data/librispeech")
    f.add_argument("-n", type=int, default=40)
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--min-s", type=float, default=3.0)
    f.add_argument("--max-s", type=float, default=12.0)
    f.add_argument("--tarball", help="use a local test-clean.tar.gz instead of downloading")
    f.set_defaults(fn=cmd_fetch)

    pr = sub.add_parser("prepare", help="apply telephony conditions to a source set")
    pr.add_argument("--source", help="folder with source.jsonl or transcripts.tsv (default: bundled sample)")
    pr.add_argument("--conditions", help="conditions YAML")
    pr.add_argument("--out", default="twb-data/prepared")
    pr.add_argument("--seed", type=int, default=1234)
    pr.add_argument("--limit", type=int, help="use only the first N utterances")
    pr.set_defaults(fn=cmd_prepare)

    r = sub.add_parser("run", help="stream a prepared set to a provider")
    r.add_argument("manifest", help="manifest.jsonl or the prepared folder")
    r.add_argument("--provider", required=True)
    r.add_argument("--opt", action="append", default=[], help="provider option key=value (repeatable)")
    r.add_argument("--out", default="twb-runs/run")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--tail", type=float, default=0.6, help="seconds of silence sent after the audio")
    r.add_argument("--no-pace", action="store_true", help="send audio as fast as possible (debug only)")
    r.add_argument("--limit", type=int)
    r.add_argument("--only", nargs="+", help="run only these condition names")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="score run folders and write JSON, Markdown and charts")
    rp.add_argument("runs", nargs="+", help="run folders written by `twb run`")
    rp.add_argument("--out", default="twb-report")
    rp.add_argument("--baseline", help="baseline condition name (default: first condition)")
    rp.add_argument("--only", nargs="+", help="report only these condition names, in this order")
    rp.add_argument("--no-charts", action="store_true")
    rp.set_defaults(fn=cmd_report)

    pv = sub.add_parser("providers", help="list providers and whether they can start here")
    pv.set_defaults(fn=cmd_providers)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
