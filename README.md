# Veritas — News Integrity Engine

A fake news detector powered by a fine-tuned BERT model, with a clean editorial UI, linguistic signal analysis, and attention-based explainability.

---

## Features

- **BERT classification** — fine-tuned `bert-base-uncased` with 97% validation accuracy
- **Linguistic signal breakdown** — detects 8 categories of language patterns (sensationalist language, unverified claims, emotional manipulation, absolute framing, punctuation abuse, excessive caps, source citations, balanced reporting)
- **Attention explainer** — shows which tokens most influenced the prediction
- **Plain-English explanation** — confidence tier, summary, and tailored recommendation per result
- **Three input modes** — paste text, submit a URL, or upload a `.txt` / `.pdf` file
- **Batch endpoints** — analyse multiple texts or URLs in a single API call

---

## Project Structure

```
fn_detector/
├── app.py                  # FastAPI application and route handlers
├── detector.py             # BERT model, signal analysis, explanation builder
├── requirements.txt
├── models/
│   └── model.safetensors   # Fine-tuned BERT weights (not committed — see below)
└── ui/
    └── index.html          # Single-file frontend
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/fn_detector.git
cd fn_detector
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the model weights

Place your fine-tuned weights at:

```
models/model.safetensors
```

The model must be a `BertForSequenceClassification` checkpoint with `num_labels=2` (label 0 = fake, label 1 = real), saved in safetensors format.

If you don't have weights yet, you can fine-tune from `bert-base-uncased` on any fake/real news dataset (e.g. [LIAR](https://huggingface.co/datasets/liar), [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)) and save with:

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "models/model.safetensors")
```

### 3. Run the server

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000).

---

## API Reference

All endpoints return JSON. Batch endpoints accept up to any number of items but performance degrades with very long lists.

### `POST /analyze/text`

```json
{ "text": "Article body here..." }
```

### `POST /analyze/url`

```json
{ "url": "https://example.com/article" }
```

Extracts article text using `newspaper3k`, then runs analysis.

### `POST /analyze/file`

Multipart form upload. Accepts `.txt` and `.pdf` files.

```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@article.pdf"
```

### `POST /analyze/batch/text`

```json
{ "items": ["Article one...", "Article two..."] }
```

### `POST /analyze/batch/url`

```json
{ "items": ["https://example.com/a", "https://example.com/b"] }
```

### Response schema

Every analysis endpoint returns:

```json
{
  "is_fake": true,
  "confidence": 0.934,
  "probability": {
    "fake": 0.934,
    "real": 0.066
  },
  "attention": [
    { "token": "shocking", "score": 1.0 },
    { "token": "exposed", "score": 0.87 }
  ],
  "signals": [
    {
      "key": "sensationalist",
      "label": "Sensationalist Language",
      "description": "Exaggerated or emotionally charged words designed to provoke reaction.",
      "positive": false,
      "count": 3,
      "examples": ["shocking", "exposed", "bombshell"]
    }
  ],
  "explanation": {
    "summary": "The model strongly classifies this as misinformation (93.4% certainty)...",
    "tier": "high",
    "tier_label": "High confidence",
    "negative_signal_count": 5,
    "positive_signal_count": 0,
    "key_tokens": ["shocking", "exposed", "they"],
    "recommendation": "Cross-check with multiple established news outlets...",
    "caveat": "This model analyses linguistic patterns only — not external facts..."
  }
}
```

URL and file endpoints additionally include:

```json
{
  "extracted_text_preview": "First 300 characters of extracted text...",
  "filename": "article.pdf"
}
```

---

## Linguistic Signal Categories

| Signal | Polarity | What it detects |
|---|---|---|
| Sensationalist Language | ⚠ Negative | Alarming or exaggerated vocabulary |
| Unverified Claims | ⚠ Negative | Vague sourcing ("sources say", "insiders claim") |
| Emotional Manipulation | ⚠ Negative | Fear/anger appeals, tribal identity language |
| Absolute / Hyperbolic Claims | ⚠ Negative | All-or-nothing framing, zero nuance |
| Excessive Capitalisation | ⚠ Negative | ALL CAPS words (4+ characters) |
| Punctuation Abuse | ⚠ Negative | Multiple `!!` or `??` sequences |
| Source Citations | ✓ Positive | Named institutions, studies, official statements |
| Balanced Reporting | ✓ Positive | Hedged language, opposing viewpoints acknowledged |

---

## Notes and Limitations

- The model analyses **linguistic patterns only** — it does not verify external facts, check sources, or access the internet.
- Satire, opinion pieces, and highly technical or translated text may score unexpectedly.
- Results near the 50% boundary (low confidence) should be treated as inconclusive — always verify independently.
- The URL extractor (`newspaper3k`) may fail on paywalled, JavaScript-rendered, or bot-protected pages.

---

## License

MIT
