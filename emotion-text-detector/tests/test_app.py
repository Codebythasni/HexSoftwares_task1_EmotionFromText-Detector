"""Tests for the Flask web interface.

These tests use Flask's built-in test client — no extra dependencies.
The existing unit tests in test_emotion_detector.py are untouched.
"""

import unittest

import app


class TestFlaskApp(unittest.TestCase):
    """Test the web routes and form handling."""

    def setUp(self):
        self.client = app.app.test_client()

    def test_get_renders_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Emotion from Text Detector", response.data)
        self.assertIn(b"Analyze Emotion", response.data)
        self.assertIn(b"Your sentence", response.data)

    def test_empty_input_shows_error(self):
        response = self.client.post("/", data={"text": "   "})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter some text", response.data)
        # No result card should be rendered for empty input.
        self.assertNotIn(b"Polarity", response.data)

    def test_happy_sentence_renders_result(self):
        response = self.client.post("/", data={"text": "I am very happy today"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Happy", response.data)
        self.assertIn("😊".encode("utf-8"), response.data)
        self.assertIn(b"Polarity", response.data)
        self.assertIn(b"Confidence", response.data)

    def test_sad_sentence_renders_result(self):
        response = self.client.post("/", data={"text": "I feel so sad and lonely"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Sad", response.data)
        self.assertIn("😢".encode("utf-8"), response.data)

    def test_angry_keyword_renders_result(self):
        response = self.client.post("/", data={"text": "I am furious about this"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Angry", response.data)
        self.assertIn("😡".encode("utf-8"), response.data)

    def test_neutral_sentence_renders_result(self):
        response = self.client.post("/", data={"text": "The weather is normal today"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Neutral", response.data)
        self.assertIn("😐".encode("utf-8"), response.data)

    def test_unexpected_error_handled_gracefully(self):
        # Simulate a crash in the detector and check no traceback leaks out.
        original = app.detect_emotion

        def explode(text):
            raise RuntimeError("boom")

        app.detect_emotion = explode
        try:
            response = self.client.post("/", data={"text": "anything"})
        finally:
            app.detect_emotion = original

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Something went wrong", response.data)
        self.assertNotIn(b"Traceback", response.data)


if __name__ == "__main__":
    unittest.main()