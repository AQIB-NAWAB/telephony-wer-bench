import numpy as np
import pytest

from twb import impair
from twb.audio import rms


def test_loss_mask_rate_and_seed():
    a = impair.loss_mask(20000, 0.1, np.random.default_rng(3))
    b = impair.loss_mask(20000, 0.1, np.random.default_rng(3))
    assert (a == b).all()
    assert abs(a.mean() - 0.1) < 0.01


def test_loss_mask_zero_and_bad_values():
    assert not impair.loss_mask(100, 0.0, np.random.default_rng(0)).any()
    with pytest.raises(ValueError):
        impair.loss_mask(10, 1.0, np.random.default_rng(0))


def test_burst_losses_cluster():
    rng = np.random.default_rng(5)
    iid = impair.loss_mask(50000, 0.1, rng)
    burst = impair.loss_mask(50000, 0.1, np.random.default_rng(5), burst=0.7)
    assert abs(burst.mean() - 0.1) < 0.02

    def mean_run(m):
        runs, cur = [], 0
        for v in m:
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        return np.mean(runs)

    assert mean_run(burst) > mean_run(iid) * 1.5


def test_plc_none_zeroes_lost_packets():
    frames = np.ones((5, 160), dtype=np.float32)
    lost = np.array([False, True, False, False, False])
    out = impair.apply_plc(frames, lost, "none")
    assert (out[1] == 0).all() and (out[0] == 1).all()


def test_plc_repeat_fades_to_silence():
    frames = np.arange(1, 8, dtype=np.float32)[:, None] * np.ones((1, 4), dtype=np.float32)
    lost = np.array([False, True, True, True, True, False, False])
    out = impair.apply_plc(frames, lost, "repeat")
    # last good packet is 1.0, gains 0.75, 0.5, 0.25 then silence
    assert out[1, 0] == pytest.approx(0.75)
    assert out[2, 0] == pytest.approx(0.5)
    assert out[3, 0] == pytest.approx(0.25)
    assert out[4, 0] == 0.0
    assert out[5, 0] == 6.0


def test_plc_unknown_mode():
    with pytest.raises(ValueError):
        impair.apply_plc(np.zeros((2, 4)), np.array([True, False]), "magic")


def test_packet_loss_keeps_length_and_reports_rate(speechish):
    audio, sr = speechish
    out, realised = impair.packet_loss(audio, 16000, 0.2, np.random.default_rng(1))
    assert len(out) == len(audio)
    assert 0.0 < realised < 0.5


def test_jitter_late_packets_are_lost():
    rng = np.random.default_rng(0)
    none = impair.jitter_loss_mask(5000, 0.0, 60, rng)
    assert not none.any()
    tight = impair.jitter_loss_mask(5000, 40, 20, np.random.default_rng(0)).mean()
    roomy = impair.jitter_loss_mask(5000, 40, 120, np.random.default_rng(0)).mean()
    assert tight > roomy


@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
def test_noise_has_unit_rms(kind):
    n = impair.noise_signal(kind, 16000, 16000, np.random.default_rng(0))
    assert len(n) == 16000
    assert rms(n) == pytest.approx(1.0, rel=1e-3)


def test_pink_noise_has_more_low_frequency_energy():
    n = impair.noise_signal("pink", 32000, 16000, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(n)) ** 2
    low, high = spec[10:200].mean(), spec[10000:15000].mean()
    assert low > high * 5


@pytest.mark.parametrize("snr", [0, 10, 20])
def test_add_noise_hits_target_snr(speechish, snr):
    audio, sr = speechish
    noise = impair.noise_signal("white", len(audio), sr, np.random.default_rng(2))
    mixed = impair.add_noise(audio, sr, noise, snr)
    added = mixed - audio
    measured = 20 * np.log10(impair.active_level(audio, sr) / rms(added))
    assert measured == pytest.approx(snr, abs=0.1)


def test_babble_signal_unit_rms():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(8000).astype(np.float32) for _ in range(3)]
    b = impair.babble_signal(clips, 20000, rng)
    assert len(b) == 20000 and rms(b) == pytest.approx(1.0, rel=1e-3)
