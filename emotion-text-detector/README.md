# Emotion from Text Detector

A beginner-friendly Python project that analyzes a sentence and classifies its
emotion as **Happy**, **Sad**, **Angry**, or **Neutral** — each with an emoji.

Built for an AI internship project (Project 02 / Phase 3). No machine learning
training and no models from scratch — it relies on the well-known, easy-to-explain
`TextBlob` library for sentiment analysis.

## Features

- Classifies text into 4 emotions with emoji labels: 😊 Happy, 😢 Sad, 😡 Angry, 😐 Neutral.
- Returns **polarity**, **subjectivity**, and a **confidence** score.
- Validates empty / whitespace-only input.
- Interactive command-line interface that accepts repeated sentences.
- **Responsive web interface** (Flask) that works on laptop and mobile browsers.
- Handles simple negation such as "I am not happy" (see *Negation Handling*).
- Friendly exit commands: `quit` or `exit`.
- Clean, modular, commented code suitable for explaining in an internship.

## Technologies

- **Python 3.8+**
- **TextBlob** — a simple NLP library built on top of NLTK.
- **Flask** — a tiny web framework for the browser interface.
- `unittest` for testing (comes with Python, no extra dependency).

## How TextBlob Polarity Works

TextBlob's `TextBlob(text).sentiment` returns two numbers:

| Name          | Range | Meaning                                                            |
| ------------- | ----- | ------------------------------------------------------------------ |
| `polarity`    | -1 to 1 | How positive (near +1) or negative (near -1) the text is.         |
| `subjectivity`| 0 to 1   | How opinion-based the text is (near 1) vs fact-based (near 0).    |

Example:

```python
from textblob import TextBlob

TextBlob("I am very happy today").sentiment.polarity      # 0.80
TextBlob("I feel so sad and lonely").sentiment.polarity   # -0.30
TextBlob("The sky is blue").sentiment.polarity            # 0.0
```

TextBlob computes these values from a lexicon — a large dictionary that maps
words to polarity/objectivity scores. The sentence score is an average over
its words.

## How Emotion Classification Works

The detector (`src/emotion_detector.py`) uses simple, explainable rules:

1. Clean and validate the input (reject empty text).
2. Compute `polarity` and `subjectivity` with TextBlob.
3. Classify with thresholds:

```
if the text has a non-negated anger keyword AND polarity <= +0.15:   -> Angry
elif polarity > 0.15:                                                -> Happy
elif polarity < -0.15:                                               -> Sad
else:                                                                -> Neutral
```

The `0.15` threshold creates a neutral "dead zone", so neutral phrasing like
"The sky is blue" isn't labelled happy just because the score is slightly above 0.

### How Anger Is Approximated

TextBlob has **no anger dimension**. To estimate anger, this project combines
two signals:

1. An **explicit anger keyword** that is **not negated** (e.g. `angry`, `mad`,
   `furious`, `hate`, `outraged`, `unfair`, ...), plus
2. A polarity that is **not strongly positive** — if the sentence is overall
   clearly positive, it wins: `"I am furious but extremely happy today"` is Happy.

So `"I am furious about this"` is **Angry**, while `"I failed my exam and I feel
miserable"` (negative, but no anger keyword) is **Sad**.

> A TextBlob quirk: several anger words (`furious`, `irritated`, `livid`,
> `rage`) have **no polarity entry** in TextBlob's lexicon and score `0.0`.
> Without the keyword rule they would be classified as Neutral, so the keyword
> rule is what catches them.

### Negation Handling (investigated, not assumed)

We tested how TextBlob actually handles negation:

| Sentence              | Polarity in TextBlob | Our output |
| --------------------- | -------------------- | ---------- |
| "I am happy"          | +0.80                | 😊 Happy   |
| "I am sad"            | -0.50                | 😢 Sad     |
| "I am not happy"      | -0.40                | 😢 Sad     |
| "I am not sad"        | +0.25                | 😊 Happy   |
| "I am not furious"    | 0.00                 | 😐 Neutral |
| "I never feel happy"  | +0.80                | 😊 Happy   |

What actually happens:

- **Direct `not <word>` negation IS handled.** TextBlob's lexicon flips the
  polarity of words like `happy`/`sad` directly. `"I am not happy"` becomes
  negative → **Sad**, and `"I am not sad"` becomes positive → **Happy**. This is
  a TextBlob feature we rely on — we did not write that logic.
- **Our own anger guard.** For anger keywords we check the few words before the
  keyword for negation words (`not`, `no`, `never`, `n't`, ...). So
  `"I am not furious"` is **not** reported as Angry (it is Neutral).
- **Not all negation is handled.** `"I never feel happy"` still scores **+0.80**
  → **Happy**, and `"I do not feel happy"` also stays **+0.80**. Double
  negatives like `"I am not not happy"` are not handled either. These are
  documented limitations, not claims of full negation support.

### Confidence (a heuristic, not a probability)

`confidence` is a simple **0–100 heuristic** that measures how strong the
available evidence is. It is **NOT a statistical probability** — the README
and CLI say so explicitly.

- **Happy / Sad:** the further polarity is from 0, the more confident.
  `confidence = |polarity| * 100`.
- **Neutral:** the closer polarity is to 0, the more confident.
  `confidence = (1 - |polarity|) * 100`.
- **Angry:** a **50% baseline**, because an explicit anger keyword is already
  strong evidence on its own; the score then rises with how negative the
  polarity is: `confidence = max(50, -polarity * 100)`. This is why
  `"I am furious about this"` (polarity 0.0) still reports **50%**, not 0%.

## Installation

1. **Install Python 3.8+** from [python.org](https://www.python.org/).
2. Clone or download this repository.
3. Create and activate a virtual environment (recommended):

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies (this installs both TextBlob and Flask):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-line interface

Run the interactive CLI from the project root:

```bash
python -m src.main
```

Type sentences one at a time. Press **Enter** with no text to be prompted again.
Type **quit** or **exit** to stop.

### Web interface (Flask)

Run the Flask app from the project root:

```bash
python app.py
```

Then open this address in your browser:

```
http://127.0.0.1:5000
```

Type a sentence, press **Analyze Emotion**, and the result card appears with
the emotion emoji, emotion name, polarity, subjectivity, and confidence. The
layout is responsive, so it also works on a mobile browser. To stop the
server, press **Ctrl+C** in the terminal.

> For development the app runs with `debug=True`. For a real deployment use a
> production server instead (e.g. `flask run` behind gunicorn/waitress).

### Using the detector in your own code

```python
from src.emotion_detector import detect_emotion

result = detect_emotion("I am very happy today")
print(result)
# {'emotion': 'Happy', 'emoji': '😊', 'polarity': 0.8,
#  'subjectivity': 1.0, 'confidence': 80.0}
```

## Web Interface (Phase 3)

A single-page browser app built with **Flask**. It does not contain any emotion
logic — it is a thin wrapper around `detect_emotion`.

**Screenshot placeholder:**

![Emotion from Text Detector web interface](screenshots/emotion_detector_web.png)

*Add a screenshot of the web page here (save it to `screenshots/`).*

### What the page shows

- Project title: **Emotion from Text Detector**
- A short description
- A text input area for the sentence
- An **Analyze Emotion** button
- A result card with:
  - Emotion emoji (😊 😢 😡 😐)
  - Emotion name (Happy, Sad, Angry, Neutral)
  - Polarity
  - Subjectivity
  - Confidence (labelled "heuristic score, not a true probability")

### Error handling on the page

- **Empty / whitespace-only input** → the page shows
  "Please enter some text before analyzing." and no result card.
- **Unexpected errors** → the page shows a friendly generic message; raw
  tracebacks are never shown to the visitor (verified by a test).

### Architecture

Flask keeps a clean separation: the browser visits `/`, submits the form via
POST, Flask passes the text to the existing detector, and renders the result.
The emotion logic lives **only** in `src/emotion_detector.py`.

```
                    +------------------------------------------------------+
                    |                    Emotion Detector                  |
                    |                                                      |
 Browser  (GET/)    |   app.py (Flask)               src/emotion_detector  |
 -------- form -->  |   index() routes /             detect_emotion(text)  |
                    |   -> detect_emotion(text)  -->  TextBlob polarity     |
                    |   -> render_template()          + keyword rules       |
  <-----------------+   templates/index.html    <---  returns dict          |
  result card       |   static/style.css styling                            |
                    +------------------------------------------------------+
```

### Project files for the web layer

```
emotion-text-detector/
├── app.py                  # Flask app (routes, form handling)
├── templates/
│   └── index.html          # page markup + result card
├── static/
│   └── style.css           # clean, responsive styling
└── src/
    ├── emotion_detector.py # UNCHANGED core logic
    └── main.py             # UNCHANGED CLI
```

## Example Inputs and Outputs

Real output from the interactive CLI:

```
=== Emotion from Text Detector ===
😊 Happy    😢 Sad    😡 Angry    😐 Neutral
Type a sentence, or "quit" / "exit" to stop.
Note: confidence is a heuristic score, not a true probability.

Enter text: I am extremely happy today

😊 Happy
--------------------------------
Polarity:     0.80
Subjectivity: 1.00
Confidence:   80.00%

Enter text: I feel lonely and miserable

😢 Sad
--------------------------------
Polarity:     -0.55
Subjectivity: 0.85
Confidence:   55.00%

Enter text: I am furious about this

😡 Angry
--------------------------------
Polarity:     0.00
Subjectivity: 0.00
Confidence:   50.00%

Enter text: The weather is normal today

😐 Neutral
--------------------------------
Polarity:     0.15
Subjectivity: 0.65
Confidence:   85.00%

Enter text: I am not happy about this

😢 Sad
--------------------------------
Polarity:     -0.40
Subjectivity: 1.00
Confidence:   40.00%

Enter text: quit
Goodbye!
```

## Running the Tests

```bash
python -m unittest discover -v
```

The **31 tests** (defined in `tests/test_emotion_detector.py` and
`tests/test_app.py`) run with plain `unittest` — no pytest needed.

- `test_emotion_detector.py` (24): happy, sad, neutral, angry, strongly
  positive, strongly negative, anger keywords (including `furious`/`irritated`),
  neutral/empty/whitespace input, negation examples (`not happy`, `not sad`,
  `not furious`), emoji mapping, and confidence bounds including the anger
  baseline.
- `test_app.py` (7): Flask route rendering, empty-input error, each of the four
  emotions through the web form, and graceful handling of an unexpected crash.

## Limitations

- **Anger is keyword-based.** TextBlob has no anger score, so anger words not
  in our list, or anger expressed without keywords (e.g. sarcasm, cursing),
  will be missed.
- **Some anger words are polarity-neutral.** `furious`, `irritated`, `livid`
  and `rage` score `0.0` in TextBlob; they are only caught by our keyword rule.
- **Negation is only partially handled.** Direct `not <word>` works; `never`,
  `"I do not feel happy"` and double negatives do not.
- **No context.** "I am not happy" is negative, but a sarcastic or humorous
  sentence is treated literally.
- **Single-language.** TextBlob works best with English.
- **Short sentences, weak signal.** Very short or ambiguous text often lands
  in Neutral.
- **Confidence is a heuristic**, not a calibrated statistical probability.
- **The web server is for local development.** `python app.py` uses Flask's
  built-in server with `debug=True`; it is not tuned for public deployment.

## Future Improvements

- Add a general negation rule (e.g. flip polarity when "never"/"do not" appears
  within a word window), replacing the `not <word>` only behavior.
- Expand the anger keyword list or use a dedicated emotion lexicon
  (e.g. NRC Emotion Lexicon).
- Support more languages (TextBlob can translate via `TextBlob(text).translate()`).
- Support longer documents with a per-sentence breakdown.
- Enhance the web interface: history of past analyses, live typing feedback,
  and a production-ready server (waitress/gunicorn).

## Project Structure

```
emotion-text-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                     # Flask web interface
├── templates/
│   └── index.html             # page markup + result card
├── static/
│   └── style.css              # clean, responsive styling
├── src/
│   ├── __init__.py
│   ├── emotion_detector.py    # core classification logic
│   └── main.py                # interactive CLI
└── tests/
    ├── __init__.py
    ├── test_emotion_detector.py   # 24 tests (detector)
    └── test_app.py                # 7 tests (Flask routes)
```