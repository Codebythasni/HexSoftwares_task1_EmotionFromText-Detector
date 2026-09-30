"""Flask web interface for the Emotion from Text Detector.

This is only a thin web layer. ALL emotion detection logic lives in
`src.emotion_detector.detect_emotion` — Flask just passes the user's
text to it and shows the result.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from flask import Flask, render_template, request

from src.emotion_detector import detect_emotion

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Show the form (GET) or analyze the submitted text (POST)."""
    result = None
    error = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            # Graceful handling for empty / whitespace-only input.
            error = "Please enter some text before analyzing."
        else:
            try:
                result = detect_emotion(text)
            except ValueError as exc:
                error = str(exc)
            except Exception:
                # Never show raw tracebacks to the visitor.
                error = "Something went wrong while analyzing your text. Please try again."

    return render_template("index.html", result=result, error=error, text=text)


if __name__ == "__main__":
    app.run(debug=True)