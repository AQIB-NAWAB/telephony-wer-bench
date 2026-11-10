import pytest

from twb.normalize import normalize, number_to_words


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", "hello world"),
        ("DON'T TROUBLE TO ANSWER", "do not trouble to answer"),
        ("I'm sure they're here", "i am sure they are here"),
        ("It can't wait; it won't wait", "it can not wait it will not wait"),
        ("well-known", "well known"),
        ("Um, so, uh, yes", "so yes"),
        ("There are 3 cats and 21 dogs", "there are three cats and twenty one dogs"),
        ("Café", "cafe"),
        ("it's [noise] fine", "it fine"),
        ("50% off", "fifty percent off"),
        ("   spaced    out   ", "spaced out"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_same_text_different_case_and_punctuation_match():
    assert normalize("THAT'S THE WAY WITH YOU") == normalize("That's the way with you.")


@pytest.mark.parametrize(
    ("n", "words"),
    [(0, "zero"), (13, "thirteen"), (40, "forty"), (99, "ninety nine"), (100, "one hundred"),
     (101, "one hundred one"), (1999, "one thousand nine hundred ninety nine"),
     (2000000, "two million")],
)
def test_number_to_words(n, words):
    assert number_to_words(n) == words
