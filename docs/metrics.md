# Metrics

All metrics are computed from the recorded events of a run (`results.jsonl`), so you can re-score old runs with a new version of the scorer.

## Events

A provider sends two kinds of events while the audio streams:

- **partial**: the current best guess for the segment that has not been finalized yet.
- **final**: a segment that will not change.

Each event has `t`, the wall-clock seconds since the first audio frame was sent, and `audio_t`, how much audio had been sent at that moment. A provider that never finalizes mid-stream (for example the simulated-streaming Whisper provider) sends one final when the stream ends.

The **hypothesis** for accuracy is all final segments joined with spaces. If a provider sent no finals, the last partial is used.

## Text normalization

Applied to the reference and the hypothesis before scoring (`src/twb/normalize.py`):

1. Unicode decomposition, accents removed.
2. Lowercase.
3. Text inside brackets or parentheses removed (for example `[noise]`).
4. `%` becomes "percent", `&` becomes "and", `$` becomes "dollars".
5. Digits spelled out as words (`21` becomes "twenty one", `3.5` becomes "three point five").
6. Contractions expanded: `won't` to "will not", `can't` to "can not", `n't` to " not", `'re`, `'ve`, `'ll`, `'m`, `'d` to full words. A trailing `'s` is dropped.
7. Remaining apostrophes removed, other punctuation (including hyphens) becomes a space.
8. Filler words dropped: um, uh, umm, uhh, hmm, mm, mhm, ah, er, erm.
9. Whitespace collapsed.

It is inspired by the Whisper English normalizer that the Open ASR Leaderboard standardizes on, but it is smaller. It does not map British to American spelling and it does not handle ordinals or currency amounts beyond the rules above. Scores are therefore not directly comparable with leaderboard numbers.

## Word error rate (WER)

Levenshtein alignment on words. `WER = (S + D + I) / N` with substitutions S, deletions D, insertions I and N reference words. WER can be above 100% when the hypothesis has many insertions.

Per condition the bench reports **pooled WER**: total errors over all utterances divided by total reference words. This weights long utterances more, the same way the Open ASR Leaderboard and Pipecat's "Pooled WER" do. Mean of per-utterance WER is not used.

**95% CI**: 1000 bootstrap resamples of utterances with a fixed seed, percentile method. With 40 utterances (about 450 reference words per condition) the interval is wide, and two conditions or providers whose intervals overlap should not be called different.

## Character error rate (CER)

Levenshtein alignment on characters of the normalized text, with spaces kept as characters. Same pooled calculation.

## Time to first partial

Seconds from the start of speech to the first event with non-empty text (partial or final). Start of speech is `speech_start_s` in the manifest, found on the clean source audio with an energy threshold. The value is clamped at 0.

This number includes the time the caller takes to say the first word, because no engine can output a word before it has heard it. Compare it between conditions and providers, not against zero.

## Final latency after end of speech

Seconds from `speech_end_s` (end of speech in the clean source) to the arrival of the last non-empty final event. Same idea as the TTFS metric in the Pipecat STT benchmark.

The bench sends `--tail` seconds of digital silence after the speech (default 0.6 s) and then closes the stream. A provider that endpoints on silence can finalize inside that window. A provider that only finalizes on close will show roughly the tail plus its flush time. Look at **flush after stream end** to separate the two: it is the time from the last frame sent to the end of `finish()`.

## Partial stability

The caller-visible transcript at each event is the finalized segments so far plus the current partial. Consecutive identical views are dropped. For every consecutive pair of views `a -> b`:

- `retracted = len(a) - common_prefix(a, b)`, the words of `a` that were shown and then changed or removed.
- **retraction rate** = total retracted words / total words shown (the sum of `len(a)` over pairs).
- **flicker rate** = pairs with `retracted > 0` / all pairs.

Words are compared after normalization. A rate of 0 means the transcript only ever grew. This definition is the bench's own, not a published standard.

## Real-time factor

Provider compute seconds divided by audio seconds, when the provider can measure its own compute (local models). Values below 1 mean the model keeps up with real time on this machine. Hosted providers report `n/a`. Because the bench paces audio in real time, wall time is always at least the audio length, so wall time divided by audio length would say nothing.

## Empty transcripts

Utterances whose hypothesis is empty. They count as all-deletions in WER, and the number is shown so a high WER caused by a provider returning nothing is visible.

## Caveats

- Latency depends on the machine, the number of worker processes sharing the CPU, and for hosted providers the network. `run.json` records the worker count and CPU count.
- With `--no-pace` the audio is sent as fast as possible. Accuracy is still valid, latency is not.
