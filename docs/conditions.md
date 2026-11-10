# Conditions

A condition is a named chain of impairments applied to clean source audio. Conditions live in a YAML file, see `examples/conditions.yaml` (the four conditions in the README) and `examples/conditions-extended.yaml` (eleven).

```yaml
conditions:
  - name: mulaw-8k-loss5
    sample_rate: 8000
    codec: mulaw
    loss: 0.05
```

## Order of operations

```
clean audio (16 kHz)
  -> noise at the source rate        (noise is in the room)
  -> resample to the condition rate  (polyphase low-pass, 8 kHz for telephony)
  -> codec round trip                (mulaw, alaw, or Opus narrowband)
  -> jitter buffer misses            (late packets become loss)
  -> random packet loss              (20 ms packets)
  -> concealment (PLC)
  -> WAV file in the manifest
```

Every random step uses a generator seeded from `sha256(base_seed | utterance id | condition name)`, so a run with the same `--seed` produces the same audio byte for byte. The seed for each file is in the manifest.

## Fields

| Field | Values | Meaning |
| --- | --- | --- |
| `name` | text | Unique name, used for folders and report rows |
| `sample_rate` | 8000, 16000 | Output sample rate |
| `codec` | `none`, `mulaw`, `alaw`, `opus` | Needs `sample_rate: 8000` when not `none` |
| `opus_bitrate` | bits per second | Default 12000. Opus runs through ffmpeg `libopus` in VoIP mode with an 8 kHz cutoff and 20 ms frames |
| `loss` | 0 to under 1 | Average share of 20 ms packets lost |
| `loss_burst` | 0 to 1 | If above 0, a two-state (Gilbert-Elliott) model is used. The value is the chance a lost packet is followed by another. The average loss stays close to `loss` |
| `plc` | `repeat`, `none` | `repeat`: repeat the last good packet with gain 0.75, 0.5, 0.25, then silence. `none`: insert silence |
| `jitter_ms` | 0 or more | Each packet gets a delay drawn from the absolute value of a normal distribution with this standard deviation |
| `jitter_buffer_ms` | default 60 | A packet delayed more than this misses its playout slot and counts as lost |
| `noise` | `{type, snr_db}` | `type`: `white`, `pink`, `brown`, `babble`. `babble` mixes four other utterances from the source set |

## What the numbers mean

- **SNR** is measured against the active speech level (the louder frames, down to 40 dB below the peak), not the average over silence. 10 dB SNR is noisy and 20 dB is a quiet office.
- **Realised loss** is stored in the manifest per file. With 40 utterances, the average for `loss: 0.05` was 5.2% and for `jitter_ms: 40` with a 60 ms buffer 13.4% (a normal variable with standard deviation 40 ms exceeds 60 ms about 13% of the time).
- The resample is a clean low-pass at 4 kHz. A real handset and trunk add band-pass shaping, level changes and sometimes clipping, which this does not model.
- G.711 is applied to int16 PCM at 8 kHz exactly as in the standard coder.

## Adding your own condition

Add an entry to a YAML file and pass it to `twb prepare --conditions`. Validation errors, such as a codec with a 16 kHz rate or noise without an SNR, are raised before any audio is written.
