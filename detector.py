import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file


class FakeNewsDetector:
    def __init__(self, model_path="./models/bert", max_length=512):
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
            'is_fake': prediction.item() == 0,   # 0 = fake, 1 = real
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

        # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
        if not outputs.attentions:
            return []

        # Stack → (num_layers, batch, num_heads, seq_len, seq_len)
        # Select batch 0 → (num_layers, num_heads, seq_len, seq_len)
        attentions = torch.stack(outputs.attentions, dim=0)[:, 0, :, :, :]

        # Mean over layers and heads → (seq_len, seq_len)
        avg_attention = attentions.mean(dim=(0, 1))

        # CLS token row: how much [CLS] attends to every other token
        cls_attention = avg_attention[0]  # (seq_len,)

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

        # Sort by score descending, take top_n
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_n]

        max_score = pairs[0][1] if pairs else 1.0
        return [
            {'token': tok, 'score': round(score / max_score, 4)}
            for tok, score in pairs
        ]

    def analyze(self, text: str) -> dict:
        """Full analysis: prediction + attention."""
        result = self.predict(text)
        result['attention'] = self.get_attention(text)
        return result
