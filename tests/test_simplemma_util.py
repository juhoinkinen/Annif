"""Unit tests for Simplemma utility functions"""

from annif.simplemma_util import get_language_detector


def test_get_language_detector():
    detector = get_language_detector("en")
    text = "She said 'au revoir' and left"
    proportion = detector.proportion_in_target_languages(text)
    assert proportion == 0.75


def test_get_language_detector_many():
    detector = get_language_detector(("en", "fr"))
    text = "She said 'au revoir' and left"
    proportion = detector.proportion_in_target_languages(text)
    assert proportion == 1.0
