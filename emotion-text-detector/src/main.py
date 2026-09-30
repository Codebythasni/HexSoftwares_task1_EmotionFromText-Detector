"""Command-line interface for the Emotion from Text Detector.

Run this file to start an interactive session where the user can type
sentences one after another. Type "quit" or "exit" to leave, or just
press Enter.
"""

import sys

from src.emotion_detector import detect_emotion

# Windows consoles may default to cp1252, which cannot print emojis.
# Force UTF-8 so the emoji labels always render (works piped and interactively).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass  # older Python / non-tty streams that do not support reconfigure

EXIT_COMMANDS = {"quit", "exit"}

SEPARATOR = "-" * 32


def run_interactive():
    """Keep asking for text until the user quits."""
    print("=== Emotion from Text Detector ===")
    print("😊 Happy    😢 Sad    😡 Angry    😐 Neutral")
    print('Type a sentence, or "quit" / "exit" to stop.')
    print("Note: confidence is a heuristic score, not a true probability.\n")

    while True:
        try:
            user_input = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Covers Ctrl+D / Ctrl+C so the loop exits cleanly.
            print("\nGoodbye!")
            return

        if not user_input:
            print("Please enter some text (empty input is ignored).\n")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            return

        try:
            result = detect_emotion(user_input)
        except ValueError as error:
            # Should not normally happen because we check for empty input
            # above, but kept for safety.
            print(f"Error: {error}\n")
            continue

        print(f"\n{result['emoji']} {result['emotion']}")
        print(SEPARATOR)
        print(f"{'Polarity:':<14}{result['polarity']:.2f}")
        print(f"{'Subjectivity:':<14}{result['subjectivity']:.2f}")
        print(f"{'Confidence:':<14}{result['confidence']:.2f}%")
        print()


if __name__ == "__main__":
    run_interactive()