# Contributing

Thanks for looking. This is a small project, so the rules are short.

## Setup

```bash
git clone https://github.com/AQIB-NAWAB/telephony-wer-bench
cd telephony-wer-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

The tests use ffmpeg for two checks (G.711 decode tables and Opus). Without ffmpeg or libopus those tests are skipped.

For the local providers: `pip install -e ".[vosk,whisper]"`.

## Adding a provider

1. Add `src/twb/providers/<name>.py` with a `Provider` and a `Session`. See `docs/architecture.md` for the contract. Stamp events with `time.monotonic()` as they arrive.
2. Register it in `src/twb/providers/__init__.py`.
3. Set `tested = False` on the class until someone has actually run it against the real service, and say so in the module docstring. Move it to tested only with a run folder or log to show for it.
4. Read keys from environment variables. Never commit keys or recordings you do not have the right to share.

## Adding a condition

Conditions are YAML, see `docs/conditions.md`. If you add a new kind of impairment, add it in `impair.py` or `codecs.py`, take a seeded `numpy.random.Generator`, and add tests for the numbers it should produce (loss rate, SNR, length preserved).

## Results

- Never invent or round up results. A chart in the README must come from a run folder in `results/`, and the caption must say how many utterances, which provider and which machine.
- If you change normalization or a metric, say so in `docs/metrics.md` and regenerate the images with `python scripts/make_images.py`.
- Latency numbers depend on the machine. Report the worker count and CPU count (they are in `run.json`).

## Style

`ruff check .` must pass. Plain wording in docs, short sentences, no hype.
