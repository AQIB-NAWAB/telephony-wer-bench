import pytest

from twb.metrics import (
    Event,
    cer,
    cumulative_views,
    edit_counts,
    final_latency,
    final_text,
    partial_stability,
    percentile,
    time_to_first_partial,
    wer,
    wer_counts,
)


def test_identical_is_zero():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_known_counts():
    c = edit_counts(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
    assert (c.substitutions, c.deletions, c.insertions, c.ref_len) == (1, 0, 1, 4)
    assert c.rate == pytest.approx(0.5)
    c = edit_counts(["a", "b", "c", "d"], ["a", "c", "d"])
    assert (c.substitutions, c.deletions, c.insertions) == (0, 1, 0)


def test_all_deleted_and_all_inserted():
    assert wer("a b c", "") == 1.0
    c = wer_counts("", "a b")
    assert c.insertions == 2 and c.rate == float("inf")
    assert wer("", "") == 0.0


def test_wer_can_exceed_one():
    assert wer("a", "b c d") == pytest.approx(3.0)


def test_wer_uses_normalizer():
    assert wer("DON'T GO", "do not go!") == 0.0


def test_cer():
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert cer("hello world", "hello  world") == 0.0


def test_counts_add_up_to_corpus_wer():
    a = wer_counts("one two three", "one two")
    b = wer_counts("four five", "four six")
    total = a + b
    assert total.errors == 2 and total.ref_len == 5


def test_time_to_first_partial():
    ev = [Event("partial", "", 0.4), Event("partial", "hello", 0.9), Event("final", "hello there", 2.0)]
    assert time_to_first_partial(ev, speech_start=0.2) == pytest.approx(0.7)
    assert time_to_first_partial([Event("partial", "  ", 1.0)], 0.0) is None
    assert time_to_first_partial([Event("partial", "hi", 0.1)], 0.5) == 0.0


def test_final_latency_uses_last_final():
    ev = [Event("final", "a b", 1.5), Event("partial", "c", 1.8), Event("final", "c", 2.4)]
    assert final_latency(ev, speech_end=2.0) == pytest.approx(0.4)
    assert final_latency([Event("partial", "x", 1.0)], 2.0) is None


def test_final_text_joins_segments_or_falls_back():
    ev = [Event("partial", "he", 0.1), Event("final", "hello", 0.5), Event("final", "world", 1.0)]
    assert final_text(ev) == "hello world"
    assert final_text([Event("partial", "a", 0.1), Event("partial", "a b", 0.2)]) == "a b"
    assert final_text([]) == ""


def test_cumulative_views_commit_finals():
    ev = [
        Event("partial", "hello", 0.1),
        Event("final", "hello there", 0.5),
        Event("partial", "how", 0.8),
        Event("final", "how are you", 1.2),
    ]
    views = cumulative_views(ev)
    assert views == [["hello"], ["hello", "there"], ["hello", "there", "how"],
                     ["hello", "there", "how", "are", "you"]]


def test_stability_no_changes():
    ev = [Event("partial", "a", 0.1), Event("partial", "a b", 0.2), Event("final", "a b c", 0.3)]
    st = partial_stability(ev)
    assert st.words_retracted == 0 and st.flicker_rate == 0.0 and st.updates == 2


def test_stability_counts_rewritten_words():
    ev = [
        Event("partial", "i scream", 0.1),
        Event("partial", "ice cream shop", 0.2),  # "i scream" -> both words rewritten (2 retracted)
        Event("final", "ice cream shop", 0.3),
    ]
    st = partial_stability(ev)
    assert st.words_shown == 2 and st.words_retracted == 2
    assert st.retraction_rate == 1.0
    assert st.flicker_rate == pytest.approx(1.0)


def test_stability_ignores_repeats():
    ev = [Event("partial", "a", 0.1), Event("partial", "a", 0.2), Event("partial", "a b", 0.3)]
    assert partial_stability(ev).updates == 1


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 50) is None
