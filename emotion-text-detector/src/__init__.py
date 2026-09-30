"""Emotion from Text Detector package.

A beginner-friendly project that classifies the emotion of a sentence
as Happy, Sad, Angry, or Neutral using TextBlob sentiment analysis.
"""

from .emotion_detector import detect_emotion

__all__ = ["detect_emotion"]