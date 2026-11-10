# Research notes: why a telephony streaming WER bench

Written 10 November 2026. Everything here was read from the linked pages on that date. Where a claim could not be confirmed, it says so.

## The problem

Voice agents on a phone line hear audio that looks nothing like the audio most speech-to-text (STT) numbers are quoted on:

- A phone call is 8 kHz audio. Narrowband telephony carries about 300 to 3400 Hz, so the high frequencies that help tell "s" from "f" are gone.
- On the PSTN and in most SIP trunks the audio is G.711, 8-bit mu-law (North America, Japan) or A-law (most other countries). It is a companding codec with about 38 dB of signal-to-quantization-noise ratio at its best, and it is not very forgiving of low levels. Defined in [ITU-T G.711](https://www.itu.int/rec/T-REC-G.711). RTP carries it as payload types 0 (PCMU) and 8 (PCMA) in 20 ms packets by default ([RFC 3551](https://www.rfc-editor.org/rfc/rfc3551)).
- Packets get lost or arrive too late for the jitter buffer. G.711 has no built-in concealment. The receiver has to invent audio, for example with the pitch-repetition method in [G.711 Appendix I](https://www.itu.int/rec/T-REC-G.711-199909-I!AppI), which works on 10 ms frames.
- Some legs use Opus at narrowband bitrates ([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716) defines narrowband as 4 kHz audio bandwidth).
- The caller is in a car, a kitchen or a street.

Then the STT result goes into a pipeline that cares about latency as much as accuracy: how soon the first partial arrives, how long after the caller stops talking the final transcript is ready, and whether partials keep rewriting words that an agent may already have acted on.

## What exists

| Project | What it measures | Audio | Streaming | Open code | Source |
| --- | --- | --- | --- | --- | --- |
| Hugging Face Open ASR Leaderboard | WER and inverse real-time factor (RTFx) for 60+ systems across 10+ datasets, with standardized normalization | Clean, read and conversational, offline | No, batch evaluation | Yes, Apache-2.0 | [paper](https://arxiv.org/abs/2510.06961), [repo](https://github.com/huggingface/open_asr_leaderboard) |
| Pipecat stt-benchmark | Semantic WER (an LLM judge counts only meaning-changing errors) and TTFS, time to final segment, from VAD stop to last final | Voice-agent turns from a public dataset, played at real-time pace | Yes | Yes | [repo](https://github.com/pipecat-ai/stt-benchmark), [site](https://www.pipecat.ai/benchmarks/speech-to-text) |
| Bluejay MISTT | English WER on 2,999 frozen clips plus 400 degraded copies (phone-line band-pass and G.711, noise at 10 to 15 dB SNR, 8 kbps Opus, 2 to 5% loss, crosstalk), robustness deltas, streaming latency | Degraded | Yes, 36 of 81 models | Scores are public. I could not confirm that the harness code is open, so treat it as a public leaderboard, not a tool you can run | [announcement](https://getbluejay.ai/blog/introducing-mistt-the-multi-condition-impairment-speech-to-text-bench) |
| LiveKit EOT Bench | End-of-turn detectors, not STT | n/a | n/a | Yes | [repo](https://github.com/livekit/eot-bench) |

Read together, the picture is: accuracy leaderboards are mostly clean audio, there is a good open streaming-latency benchmark on clean-ish audio (Pipecat), and there is a public leaderboard for degraded audio (MISTT). What I could not find is an open tool you can run yourself that does all of this at once:

1. Takes your own audio and applies telephony conditions reproducibly (seeded, with a manifest), so you can test the codec, loss rate and noise level of your own line.
2. Streams the result at real-time pace in 20 ms frames to any provider, including a local model.
3. Reports classic WER and CER per condition, plus time to first partial, final latency, and how much partials are rewritten.

That is the scope of this project. It is not a leaderboard and it does not claim to rank vendors. The bundled results are from two small local models on 40 LibriSpeech utterances, and the README says so next to every chart.

## Design choices and why

**Impair the audio, do not change the model.** Every condition is produced from the same clean source utterance, so the change in WER between conditions is on the same words. MISTT pairs degraded and clean copies for the same reason.

**Order of the chain.** Noise is added first, at the source sample rate, because noise is in the room, not on the wire. Then the audio is resampled to 8 kHz with a polyphase low-pass, then run through the codec, then packet loss and jitter are applied and concealed. This is the order a real call goes through. See [conditions.md](conditions.md).

**G.711 written from the spec, checked against ffmpeg.** The coder is a numpy port of the standard segment algorithm. The decoder tables match ffmpeg's `pcm_mulaw` and `pcm_alaw` exactly for all 256 codes. The encoders agree on most inputs. They differ by one quantization step on some inputs near segment boundaries because the two implementations round the 14-bit and 13-bit linear value differently. The tests cover this.

**Simple PLC, on purpose.** The default concealment repeats the last good 20 ms packet with a falling gain and fades to silence after three lost packets. A real receiver would use something better, such as G.711 Appendix I, and the STT may behave differently on better concealment. The `plc: none` option inserts silence instead, which is the pessimistic bound. Treat the loss results as "simple concealment", not "any real network".

**Real-time pacing with absolute deadlines.** Frame `i` is released at `start + i * 20 ms`. If the consumer falls behind, frames go out at once until it catches up. This is what a live call looks like to a server. AssemblyAI's streaming docs also warn that sending chunks in a tight loop "can produce inconsistent Turn messages", which is why the bench has no fast mode for latency numbers (there is `--no-pace` for debugging only).

**Latency reference points come from the clean audio.** Speech start and end are found with a simple energy threshold on the clean source and stored in the manifest, so every condition is measured against the same two points. Noise would make a VAD on the degraded file disagree with itself between conditions.

**Final latency is measured from end of speech to the last final transcript, the same shape as Pipecat's TTFS.** The bench sends 0.6 s of silence after the audio (configurable) so endpointing engines can finalize on their own. A provider that only finalizes when the stream is closed will show about the tail length plus its flush time. The flush time is reported separately.

**Partial stability is its own metric.** An agent that acts on partials, for example to start an LLM call early, pays for every word that is later rewritten. See [metrics.md](metrics.md) for the exact definition. I did not find a standard definition for it, so this one is simple and documented, not borrowed.

**Normalization.** The Open ASR Leaderboard standardizes text normalization because WER differences from casing, punctuation, numbers and contractions are large and unrelated to recognition quality. This project uses its own smaller English normalizer, applied to both sides. It does not match the leaderboard's normalizer rule for rule, so do not compare the numbers here directly with leaderboard numbers.

## Data

The bundled sample and the default benchmark use [LibriSpeech](https://www.openslr.org/12) test-clean (Panayotov, Chen, Povey, Khudanpur, ICASSP 2015), licensed CC BY 4.0. It is read English from audiobooks, which is easier than conversational phone speech. A low WER on LibriSpeech after telephony conditions says the chain is working, not that a model will do well on real calls. The `twb prepare --source` option takes any folder with audio files and a `transcripts.tsv`, so you can test your own recordings.

## What this does not cover yet

- Real phone recordings. Everything is simulated from clean audio.
- Crosstalk (a second talker). `babble` noise is a rough stand-in.
- Acoustic echo, clipping, AGC and the band-pass and level shaping of real handsets (ITU-T P.48 and P.830 style filters). The 8 kHz resample is a clean low-pass.
- Accents, languages other than English, numbers and names as separate scores.
- Cloud providers. The Deepgram, AssemblyAI, OpenAI Realtime, Google and Azure adapters were written from public documentation and have never been run. See the README table.

## Sources

- ITU-T G.711, Pulse code modulation of voice frequencies. https://www.itu.int/rec/T-REC-G.711
- ITU-T G.711 Appendix I, packet loss concealment. https://www.itu.int/rec/T-REC-G.711-199909-I!AppI
- IETF RFC 3551, RTP profile for audio and video conferences (payload types PCMU 0 and PCMA 8). https://www.rfc-editor.org/rfc/rfc3551
- IETF RFC 6716, Definition of the Opus audio codec. https://www.rfc-editor.org/rfc/rfc6716
- Srivastav et al., Open ASR Leaderboard. https://arxiv.org/abs/2510.06961 and https://github.com/huggingface/open_asr_leaderboard
- Pipecat STT benchmark, TTFS definition. https://github.com/pipecat-ai/stt-benchmark/blob/main/docs/measuring-ttfs.md
- Bluejay Labs, MISTT announcement, 28 October 2026. https://getbluejay.ai/blog/introducing-mistt-the-multi-condition-impairment-speech-to-text-bench
- LibriSpeech corpus. https://www.openslr.org/12
- AssemblyAI streaming API reference (pacing advice, encodings). https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api
- Deepgram live streaming reference. https://developers.deepgram.com/reference/speech-to-text/listen-streaming
- OpenAI realtime transcription guide. https://developers.openai.com/api/docs/guides/realtime-transcription
- Vosk. https://alphacephei.com/vosk/ and faster-whisper. https://github.com/SYSTRAN/faster-whisper
