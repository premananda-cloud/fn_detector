import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Linguistic signal patterns for explainability
# ---------------------------------------------------------------------------

SIGNAL_PATTERNS = {
    "sensationalist": {
        "label": "Sensationalist Language",
        "description": "Exaggerated, alarming, or emotionally charged words designed to provoke reaction.",
        "patterns": [
            r"\b(shocking|bombshell|explosive|outrage|scandal|exposed|revealed|unbelievable|"
            r"jaw-dropping|stunning|alarming|catastrophic|devastating|unprecedented|"
            r"terrifying|horrifying|disgusting|insane|crazy|unreal|mindblowing)\b"
        ],
    },
    "hedging": {
        "label": "Unverified Claims",
        "description": "Vague sourcing language that avoids attribution to named, credible sources.",
        "patterns": [
            r"\b(sources say|insiders claim|reportedly|allegedly|rumored|word is|"
            r"some people say|many believe|experts warn|officials claim|it is said|"
            r"according to insiders|anonymous sources|they don't want you to know|"
            r"mainstream media won't|what they're hiding)\b"
        ],
    },
    "emotional_appeal": {
        "label": "Emotional Manipulation",
        "description": "Appeals to fear, anger, or tribal identity rather than evidence.",
        "patterns": [
            r"\b(wake up|sheeple|patriots|traitors|evil|corrupt|destroy|"
            r"fight back|rise up|they hate us|our children|protect your family|"
            r"before it's too late|don't let them|stand up|regime|tyranny|"
            r"freedom fighters|deep state|globalists|elites)\b"
        ],
    },
    "absolute_language": {
        "label": "Absolute / Hyperbolic Claims",
        "description": "All-or-nothing framing with no nuance — a common marker of propaganda.",
        "patterns": [
            r"\b(always|never|everyone knows|nobody|no one|100%|completely|totally|"
            r"absolutely|the truth is|the fact is|undeniable|irrefutable|"
            r"proven beyond|without a doubt|definitively|once and for all)\b"
        ],
    },
    "citation_present": {
        "label": "Source Citations",
        "description": "References to studies, institutions, or named individuals — a positive credibility signal.",
        "patterns": [
            r"\b(according to|cited by|published in|study by|research from|"
            r"university|institute|journal|professor|dr\.|ph\.d|spokesperson|"
            r"said in a statement|press release|official report|data shows)\b"
        ],
        "positive": True,
    },
    "balanced_language": {
        "label": "Balanced Reporting",
        "description": "Use of hedged, measured language typical of professional journalism.",
        "patterns": [
            r"\b(however|nevertheless|on the other hand|while|although|"
            r"it is unclear|remains to be seen|could not be independently verified|"
            r"did not respond to requests for comment|declined to comment|"
            r"disputed by|contradicted by|experts disagree)\b"
        ],
        "positive": True,
    },
    "all_caps": {
        "label": "Excessive Capitalisation",
        "description": "Shouting in text — a stylistic marker of low-credibility content.",
        "patterns": [r"\b[A-Z]{4,}\b"],
    },
    "punctuation_abuse": {
        "label": "Punctuation Abuse",
        "description": "Multiple exclamation marks or question marks — hallmark of tabloid and disinformation content.",
        "patterns": [r"[!?]{2,}"],
    },
}


def analyse_signals(text: str) -> list[dict]:
    """
    Scan text for each signal category. Returns list of signal dicts
    with matched excerpt examples, counts, and polarity.
    """
    results = []

    for key, cfg in SIGNAL_PATTERNS.items():
        matches = []
        for pattern in cfg["patterns"]:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)

        if not matches:
            continue

        seen = {}
        for m in matches:
            seen[m.lower()] = seen.get(m.lower(), 0) + 1
        top = sorted(seen.items(), key=lambda x: -x[1])[:5]

        results.append({
            "key": key,
            "label": cfg["label"],
            "description": cfg["description"],
            "positive": cfg.get("positive", False),
            "count": sum(seen.values()),
            "examples": [t for t, _ in top],
        })

    results.sort(key=lambda x: (x["positive"], -x["count"]))
    return results


def build_explanation(
    is_fake: bool,
    confidence: float,
    signals: list[dict],
    attention_tokens: list[dict],
) -> dict:
    """
    Produce a structured, human-readable explanation of the prediction.
    """
    neg_signals = [s for s in signals if not s["positive"]]
    pos_signals = [s for s in signals if s["positive"]]

    neg_count = sum(s["count"] for s in neg_signals)
    pos_count = sum(s["count"] for s in pos_signals)

    if confidence >= 0.88:
        tier = "high"
        tier_label = "High confidence"
    elif confidence >= 0.65:
        tier = "moderate"
        tier_label = "Moderate confidence"
    else:
        tier = "low"
        tier_label = "Low confidence — borderline result"

    if is_fake:
        if tier == "high":
            summary = (
                f"The model strongly classifies this article as misinformation "
                f"({confidence*100:.1f}% certainty). "
                f"It detected {neg_count} negative linguistic signal(s) across "
                f"{len(neg_signals)} categor{'y' if len(neg_signals)==1 else 'ies'}."
            )
        elif tier == "moderate":
            summary = (
                f"The model leans toward misinformation ({confidence*100:.1f}% certainty) "
                f"but is not fully certain. This may reflect a mix of real and misleading content, "
                f"opinion writing, or satire."
            )
        else:
            summary = (
                f"The model nudges toward misinformation ({confidence*100:.1f}% certainty) "
                f"but sits close to the decision boundary. This result alone is inconclusive — "
                f"independent verification is essential."
            )
    else:
        if tier == "high":
            summary = (
                f"The model strongly classifies this as credible journalism "
                f"({confidence*100:.1f}% certainty). "
                f"Language patterns are consistent with factual, measured reporting."
            )
        elif tier == "moderate":
            summary = (
                f"The model leans toward credible content ({confidence*100:.1f}% certainty). "
                f"Some stylistic elements introduced uncertainty — this can occur with opinion "
                f"pieces, advocacy writing, or emotionally charged but legitimate journalism."
            )
        else:
            summary = (
                f"The model marginally favours credibility ({confidence*100:.1f}% certainty) "
                f"but the result is near the boundary. Highly specialised text, unusual "
                f"writing styles, or translated articles can produce this."
            )

    top_tokens = [
        t["token"].lstrip("##")
        for t in attention_tokens[:8]
        if t["token"] not in {"[CLS]", "[SEP]", "[PAD]"}
    ]

    if is_fake and tier == "high":
        recommendation = (
            "Cross-check with multiple established news outlets before sharing or acting on "
            "this content. Verify the publication, author, and original sources cited."
        )
    elif is_fake:
        recommendation = (
            "Treat as a yellow flag. Look for primary sources, check if claims appear in "
            "credible outlets, and consider whether this could be satire or commentary."
        )
    elif not is_fake and tier == "high":
        recommendation = (
            "Even credible-looking content can contain factual errors or bias. "
            "Verify specific claims against primary sources for consequential decisions."
        )
    else:
        recommendation = (
            "Verify the source's reputation and check whether core claims are independently "
            "corroborated before drawing conclusions."
        )

    caveat = (
        "This model was fine-tuned on BERT-base-uncased with 97% validation accuracy. "
        "It analyses linguistic patterns only — not external facts. "
        "Satire, opinion, and highly technical text may score differently."
    )

    return {
        "summary": summary,
        "tier": tier,
        "tier_label": tier_label,
        "negative_signal_count": neg_count,
        "positive_signal_count": pos_count,
        "key_tokens": top_tokens,
        "recommendation": recommendation,
        "caveat": caveat,
    }


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class FakeNewsDetector:
    def __init__(self, model_path="./models", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )

        safetensors_path = f"{model_path}/model.safetensors"
        state_dict = load_file(safetensors_path)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Predict if news is fake (0) or real (1)"""
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = probabilities[0][prediction].item()

        return {
            'is_fake': prediction.item() == 0,
            'confidence': confidence,
            'probability': {
                'real': probabilities[0][1].item(),
                'fake': probabilities[0][0].item(),
            }
        }

    def get_attention(self, text: str, top_n: int = 20) -> list[dict]:
        """
        Return top_n tokens with their averaged attention scores.
        Scores are normalised to [0, 1] relative to the top token.
        Special tokens ([CLS], [SEP], [PAD]) are excluded.
        """
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                token_type_ids=inputs.get('token_type_ids'),
                output_attentions=True,
            )

        if not outputs.attentions:
            return []

        attentions = torch.stack(outputs.attentions, dim=0)[:, 0, :, :, :]
        avg_attention = attentions.mean(dim=(0, 1))
        cls_attention = avg_attention[0]

        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0].tolist()
        )
        attention_scores = cls_attention.cpu().tolist()

        special = {'[CLS]', '[SEP]', '[PAD]'}
        pairs = [
            (tok, score)
            for tok, score in zip(tokens, attention_scores)
            if tok not in special
        ]

        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_n]

        max_score = pairs[0][1] if pairs else 1.0
        return [
            {'token': tok, 'score': round(score / max_score, 4)}
            for tok, score in pairs
        ]

    def analyze(self, text: str) -> dict:
        """Full analysis: prediction + attention + linguistic signals + explanation."""
        result = self.predict(text)
        result['attention'] = self.get_attention(text)
        result['signals'] = analyse_signals(text)
        result['explanation'] = build_explanation(
            is_fake=result['is_fake'],
            confidence=result['confidence'],
            signals=result['signals'],
            attention_tokens=result['attention'],
        )
        return result
