"""Core logic for classifying the emotion of a sentence.

This module is intentionally kept simple so an internship student can
explain every line. It uses `TextBlob` to get two numbers:

- polarity: how positive (close to +1) or negative (close to -1) a text is.
- subjectivity: how much the text is based on opinion (close to 1)
  versus fact (close to 0).

TextBlob has no "anger" dimension, so we approximate "Angry" as:
a NON-negated anger keyword (such as "angry", "mad", "furious", "hate")
combined with a polarity that is not strongly positive.

Simple negation such as "I am not happy" is already flipped to a negative
polarity by TextBlob's built-in lexicon, so our detector handles it for free.
That flip is a TextBlob feature, NOT code we write; it only works for direct
"not <word>" patterns (e.g. "I never feel happy" is NOT flipped).
"""

from textblob import TextBlob

# Keyword list used to approximate anger. TextBlob has no "anger" score, so
# we combine these explicit anger words with the overall sentiment.
# Note: several of these words ("furious", "irritated", "livid", "rage")
# have NO polarity entry in TextBlob's lexicon and score 0.0, which is why
# we detect them with this keyword rule instead of polarity alone.
ANGER_KEYWORDS = [
    "angry",
    "anger",
    "mad",
    "furious",
    "rage",
    "hate",
    "hated",
    "hates",
    "annoyed",
    "annoying",
    "frustrated",
    "frustrating",
    "irritated",
    "irritating",
    "outraged",
    "livid",
    "unfair",
]

# Words used to notice that an anger keyword is negated, so that "not furious"
# is not reported as Angry. We also treat any word ending in "n't" as negation.
NEGATION_WORDS = {"no", "not", "never", "nope", "nor", "nothing", "nowhere"}

# How many words we look back before an anger keyword to spot a negation.
NEGATION_WINDOW = 3

# Polarity values below -POLARITY_THRESHOLD count as negative, above as
# positive. Everything in between is considered neutral. This window avoids
# tiny ripple effects from words like "the" or "and" being labelled happy/sad.
POLARITY_THRESHOLD = 0.15

# Emoji shown next to each emotion in the command-line output.
EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😡",
    "Neutral": "😐",
}


def _clean_text(text):
    """Strip the text and collapse repeated whitespace to a single space."""
    return " ".join(str(text).strip().split())


def _is_negation(word):
    """Heuristic: True when a word carries a negating meaning."""
    lowered = word.lower()
    return lowered in NEGATION_WORDS or lowered.endswith("n't")


def _has_anger_keyword(text):
    """Return True when the text contains an anger keyword that is NOT negated.

    We look at up to NEGATION_WINDOW words before each keyword for a negation
    word such as "not". For example "I am not furious about this" still
    contains the word "furious", but it is negated, so this returns False.
    """
    tokens = text.split()  # text is already cleaned into single-spaced words
    for index, token in enumerate(tokens):
        if token.lower() not in ANGER_KEYWORDS:
            continue
        # Look back a few words to catch "not furious" / "don't be mad".
        window = tokens[max(0, index - NEGATION_WINDOW):index]
        if any(_is_negation(word) for word in window):
            continue  # this anger keyword is negated, skip it
        return True
    return False


def _calculate_confidence(emotion, polarity):
    """Heuristic confidence score from 0 to 100.

    IMPORTANT: this is NOT a true probability. It only measures how strong
    the available evidence (polarity / keyword) is in favour of the chosen
    emotion:

    - Happy / Sad: the further polarity is from 0, the more confident.
    - Neutral: the closer polarity is to 0, the more confident.
    - Angry: a 50% baseline because an explicit anger keyword is already
      strong evidence on its own; the score then rises with how negative
      the polarity is.
    """
    if emotion == "Angry":
        baseline = 50.0  # an explicit anger keyword is strong evidence alone
        negative_strength = min(max(-polarity, 0.0) * 100, 100.0)
        return round(max(baseline, negative_strength), 2)
    if emotion == "Neutral":
        return round((1.0 - abs(polarity)) * 100, 2)
    # Happy -> positive polarity, Sad -> negative polarity.
    return round(abs(polarity) * 100, 2)


def detect_emotion(text):
    """Classify the emotion of a sentence.

    Args:
        text (str): The sentence to analyze.

    Returns:
        dict: A dictionary with the emotion, emoji, polarity, subjectivity,
              and confidence (0-100).

    Raises:
        ValueError: If the input is empty or only whitespace.
    """
    cleaned = _clean_text(text)

    # Graceful validation: reject empty / whitespace-only input.
    if not cleaned:
        raise ValueError("Input text cannot be empty.")

    # TextBlob returns polarity in [-1, 1] and subjectivity in [0, 1].
    blob = TextBlob(cleaned)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    # Anger needs a real, non-negated anger word. The overall sentiment must
    # also not be strongly positive, so "I am furious but happy" stays Happy.
    if _has_anger_keyword(cleaned) and polarity <= POLARITY_THRESHOLD:
        emotion = "Angry"
    elif polarity > POLARITY_THRESHOLD:
        emotion = "Happy"
    elif polarity < -POLARITY_THRESHOLD:
        emotion = "Sad"
    else:
        emotion = "Neutral"

    return {
        "emotion": emotion,
        "emoji": EMOTION_EMOJIS[emotion],
        "polarity": polarity,
        "subjectivity": subjectivity,
        "confidence": _calculate_confidence(emotion, polarity),
    }