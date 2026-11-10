import json

import numpy as np
import pytest

from twb.audio import FRAME_MS, frame_samples, iter_frames, read_audio, speech_bounds, write_wav
from twb.providers.base import RawEvent, Session
from twb.runner import pace_frames, stream_utterance
from twb.streamresample import Upsampler2x


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_frame_sizes():
    assert frame_samples(8000) == 160
    assert frame_samples(16000) == 320


def test_iter_frames_pads_last():
    frames = list(iter_frames(np.ones(500, dtype=np.float32) * 0.1, 8000))
    assert [len(f) for f in frames] == [160, 160, 160, 160]
    assert frames[-1][-1] == 0


def test_pacing_releases_one_frame_per_20ms():
    clk = FakeClock()
    frames = [np.zeros(160, dtype=np.int16)] * 6
    out = list(pace_frames(frames, FRAME_MS, True, clk, clk.sleep))
    times = [round(t, 3) for _, _, t in out]
    assert times == [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


def test_pacing_does_not_drift_when_consumer_is_slow():
    clk = FakeClock()
    frames = [np.zeros(160, dtype=np.int16)] * 5
    seen = []
    for i, _, t in pace_frames(frames, FRAME_MS, True, clk, clk.sleep):
        seen.append(round(t, 3))
        if i == 1:
            clk.t += 0.05  # consumer stalls for 50 ms
    # frame 2 and 3 are already late, so they go out at once, then pacing catches up to the 80 ms mark
    assert seen[:2] == [0.0, 0.02]
    assert seen[2] == pytest.approx(0.07, abs=1e-6)
    assert seen[4] == pytest.approx(0.08, abs=1e-6)


def test_no_pace_never_sleeps():
    clk = FakeClock()
    list(pace_frames([np.zeros(160, dtype=np.int16)] * 5, FRAME_MS, False, clk, clk.sleep))
    assert clk.sleeps == []


class RecordingSession(Session):
    def __init__(self, clock):
        self.clock = clock
        self.frames = 0
        self.pending = []

    def send(self, pcm):
        self.frames += 1
        if self.frames == 10:
            self.pending.append(RawEvent("partial", "hello", self.clock()))

    def poll(self):
        out, self.pending = self.pending, []
        return out

    def finish(self):
        return [RawEvent("final", "hello world", self.clock())]


def test_stream_utterance_with_fake_clock():
    clk = FakeClock()
    audio = np.full(8000, 0.1, dtype=np.float32)  # 1 s at 8 kHz = 50 frames
    sess = RecordingSession(clk)
    res = stream_utterance(sess, audio, 8000, tail_s=0.2, pace=True, clock=clk, sleep=clk.sleep)
    assert sess.frames == 50 + 10
    assert res.audio_s == 1.0 and res.tail_s == pytest.approx(0.2)
    partial = res.events[0]
    assert partial["kind"] == "partial" and partial["t"] == pytest.approx(0.18, abs=1e-6)
    assert res.events[-1]["kind"] == "final"
    # the stream took 59 frame gaps plus the last frame: about 1.18 s
    assert res.stream_end_t == pytest.approx(1.18, abs=1e-6)


def test_speech_bounds(speechish):
    audio, sr = speechish
    start, end = speech_bounds(audio, sr)
    assert 0.2 <= start <= 0.3 and 1.2 <= end <= 1.3


def test_wav_roundtrip(tmp_path, speechish):
    audio, sr = speechish
    write_wav(tmp_path / "a.wav", audio, sr)
    back, sr2 = read_audio(tmp_path / "a.wav")
    assert sr2 == sr and np.abs(back - audio).max() < 1e-3


def test_upsampler_preserves_a_tone():
    sr = 8000
    t = np.arange(8000) / sr
    x = (0.4 * np.sin(2 * np.pi * 500 * t) * 32767).astype(np.int16)
    up = Upsampler2x()
    out = np.concatenate([up.process(x[i : i + 160]) for i in range(0, len(x), 160)])
    assert len(out) == 16000
    spec = np.abs(np.fft.rfft(out[2000:14000].astype(float)))
    freq = np.argmax(spec) * 16000 / 12000
    assert freq == pytest.approx(500, abs=5)
    # amplitude is kept
    assert np.abs(out[2000:14000]).max() == pytest.approx(0.4 * 32767, rel=0.1)


def test_json_roundtrip_of_events():
    from twb.metrics import Event

    e = Event("partial", "x", 1.23456, 0.5)
    assert Event.from_dict(json.loads(json.dumps(e.to_dict()))).text == "x"
