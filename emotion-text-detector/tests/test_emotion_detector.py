"""Unit tests for the emotion detector.

Run from the project root with:
    python -m tests.test_emotion_detector
or:
    python -m unittest discover -v
"""

import unittest

from src.emotion_detector import detect_emotion


class TestDetectEmotion(unittest.TestCase):
    """Test the detect_emotion function for each emotion class."""

    def test_happy_sentence(self):
        result = detect_emotion("I am very happy today")
        self.assertEqual(result["emotion"], "Happy")
        self.assertGreater(result["polarity"], 0.15)
        self.assertIn("polarity", result)
        self.assertIn("subjectivity", result)
        self.assertIn("confidence", result)

    def test_sad_sentence(self):
        result = detect_emotion("I feel so sad and lonely")
        self.assertEqual(result["emotion"], "Sad")
        self.assertLess(result["polarity"], -0.15)

    def test_neutral_sentence(self):
        result = detect_emotion("The sky is blue and the grass is green")
        self.assertEqual(result["emotion"], "Neutral")
        self.assertTrue(-0.15 <= result["polarity"] <= 0.15)

    def test_angry_sentence(self):
        result = detect_emotion("I am so angry at this stupid service")
        self.assertEqual(result["emotion"], "Angry")

    def test_angry_keyword_variation(self):
        result = detect_emotion("This is really unfair and frustrating")
        self.assertEqual(result["emotion"], "Angry")

    def test_negative_without_keyword_is_sad(self):
        # Negative sentiment but no anger keyword -> should be Sad, not Angry.
        result = detect_emotion("I failed my exam and I feel miserable")
        self.assertEqual(result["emotion"], "Sad")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            detect_emotion("")

    def test_whitespace_only_input_raises(self):
        with self.assertRaises(ValueError):
            detect_emotion("     ")

    def test_non_string_input(self):
        # Numbers should be handled gracefully (converted to text).
        result = detect_emotion("12345")
        self.assertEqual(result["emotion"], "Neutral")

    def test_confidence_is_bounded(self):
        cases = [
            "I am the happiest person alive",
            "I feel so sad and lonely",
            "I am so angry at this stupid service",
            "The sky is blue",
        ]
        for text in cases:
            result = detect_emotion(text)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 100.0)


class TestStrongSentiment(unittest.TestCase):
    """Strongly positive and strongly negative texts."""

    def test_strongly_positive_text(self):
        result = detect_emotion("I am extremely happy today")
        self.assertEqual(result["emotion"], "Happy")
        self.assertGreater(result["polarity"], 0.15)

    def test_strongly_negative_text(self):
        result = detect_emotion("I feel lonely and miserable")
        self.assertEqual(result["emotion"], "Sad")
        self.assertLess(result["polarity"], -0.15)


class TestAngerKeywords(unittest.TestCase):
    """Anger words whose polarity TextBlob's lexicon cannot judge."""

    def test_furious_is_angry(self):
        # "furious" has 0.0 polarity in TextBlob, so only our keyword rule
        # can catch it. Without it this would be Neutral.
        result = detect_emotion("I am furious about this")
        self.assertEqual(result["emotion"], "Angry")

    def test_irritated_is_angry(self):
        result = detect_emotion("This noise is so irritating")
        self.assertEqual(result["emotion"], "Angry")

    def test_positive_sentiment_wins_over_keyword(self):
        # Overall strongly positive sentiment should override the anger word.
        result = detect_emotion("I am furious but extremely happy today")
        self.assertEqual(result["emotion"], "Happy")

    def test_anger_detected_from_keyword_alone(self):
        result = detect_emotion("angry furious mad")
        self.assertEqual(result["emotion"], "Angry")

    def test_anger_confidence_has_keyword_baseline(self):
        # Explicit anger keywords should never report 0% confidence.
        result = detect_emotion("I am furious about this")
        self.assertGreaterEqual(result["confidence"], 50.0)


class TestNegation(unittest.TestCase):
    """What the implementation actually does with negated sentences.

    TextBlob's lexicon flips polarity for direct "not <word>" patterns, so
    "I am not happy" becomes negative and "I am not sad" becomes positive.
    That flip is a TextBlob feature we rely on; it is not code we wrote.

    Negation forms TextBlob does NOT model ("never", "I do not feel happy",
    double negatives) keep their plain polarity and are documented as such.
    """

    def test_not_happy_becomes_sad(self):
        result = detect_emotion("I am not happy")
        self.assertEqual(result["emotion"], "Sad")

    def test_not_sad_becomes_happy(self):
        result = detect_emotion("I am not sad")
        self.assertEqual(result["emotion"], "Happy")

    def test_negated_anger_keyword_is_not_angry(self):
        # Our own guard prevents "not furious" from being reported as Angry.
        result = detect_emotion("I am not furious about this")
        self.assertEqual(result["emotion"], "Neutral")

    def test_negated_anger_keyword_variation(self):
        result = detect_emotion("I am not mad at all")
        self.assertEqual(result["emotion"], "Happy")

    def test_never_is_not_handled(self):
        # Documents a known limitation: TextBlob does not flip "never".
        # "I never feel happy" still scores positive -> Happy.
        result = detect_emotion("I never feel happy")
        self.assertEqual(result["emotion"], "Happy")


class TestEmojis(unittest.TestCase):
    """Every emotion must come with its emoji label."""

    def test_emoji_mapping(self):
        expected = {
            "Happy": "😊",
            "Sad": "😢",
            "Angry": "😡",
            "Neutral": "😐",
        }
        for text, emotion in [
            ("I am very happy today", "Happy"),
            ("I feel so sad and lonely", "Sad"),
            ("I am furious about this", "Angry"),
            ("The sky is blue", "Neutral"),
        ]:
            result = detect_emotion(text)
            self.assertEqual(result["emotion"], emotion)
            self.assertEqual(result["emoji"], expected[emotion])

    def test_emoji_key_present(self):
        result = detect_emotion("I am very happy today")
        self.assertIn("emoji", result)


if __name__ == "__main__":
    unittest.main()