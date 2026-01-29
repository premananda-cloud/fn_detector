# Complete Usage Guide

## 🚀 Getting Started (Choose Your Method)

### Method 1: Demo Mode (Easiest!)
Perfect for first-time users. No arguments needed.

```bash
python demo.py
```

This will:
- Start interactive mode where you can type text
- Or type `demo` to see sample analyses
- Or type `help` for commands

### Method 2: Quick Test
Verify everything works:

```bash
python test.py
```

This will test all models and show you if they're working correctly.

### Method 3: Menu System
Interactive menu with all options:

```bash
bash run.sh
```

Choose from:
1. Demo mode
2. Interactive mode
3. System test
4. Examples
5. Custom text input
6. Exit

### Method 4: Command Line (Full Power)
For advanced usage with all features:

```bash
python main.py --text "Your news text here"
```

---

## 📚 Detailed Command Reference

### Basic Commands

#### Single Text Analysis
```bash
python main.py --text "Your text here"
python main.py -t "Your text here"        # Short form
```

#### From File
```bash
python main.py --file article.txt
python main.py -f article.txt             # Short form
```

#### Batch Processing
```bash
# Create file with one article per line
echo "Article 1 text" > articles.txt
echo "Article 2 text" >> articles.txt

python main.py --batch articles.txt
python main.py -b articles.txt            # Short form
```

#### Interactive Mode
```bash
python main.py --interactive
python main.py -i                         # Short form
```

---

## ⚙️ Configuration Options

### Model Selection

By default, all three models are used. You can disable any:

```bash
# Use only BERT
python main.py -t "text" --no-roberta --no-tfidf

# Use only RoBERTa
python main.py -t "text" --no-bert --no-tfidf

# Use only TF-IDF
python main.py -t "text" --no-bert --no-roberta

# Use BERT + RoBERTa (no TF-IDF)
python main.py -t "text" --no-tfidf
```

### Ensemble Methods

Choose how models are combined:

```bash
# Confidence weighted (default) - weights by model confidence
python main.py -t "text" --ensemble confidence_weighted

# Majority vote - simple voting
python main.py -t "text" --ensemble majority_vote
python main.py -t "text" -e majority_vote           # Short form

# Weighted average - uses predefined model weights
python main.py -t "text" --ensemble weighted_average

# Unanimous - requires all models to agree
python main.py -t "text" --ensemble unanimous

# Adaptive - chooses strategy based on agreement
python main.py -t "text" --ensemble adaptive
```

### Output Options

```bash
# Simple output (less detail)
python main.py -t "text" --simple
python main.py -t "text" -s                         # Short form

# No explanations (faster)
python main.py -t "text" --no-explain

# Quiet mode (minimal output)
python main.py -t "text" --quiet
python main.py -t "text" -q                         # Short form

# Combine options
python main.py -t "text" -s -q --no-explain
```

---

## 💡 Common Use Cases

### Use Case 1: Quick Check
You see a suspicious article and want to quickly check it:

```bash
python demo.py
# Then type or paste the text when prompted
```

### Use Case 2: Analyze Saved Article
You have an article saved in a file:

```bash
python main.py --file suspicious_article.txt
```

### Use Case 3: Check Multiple Articles
You have several articles to check:

```bash
# Put all articles in a file (one per line)
python main.py --batch all_articles.txt
```

### Use Case 4: Compare Different Models
See how different models perform:

```bash
# All models
python main.py -t "Your text"

# Only BERT
python main.py -t "Your text" --no-roberta --no-tfidf

# Only RoBERTa
python main.py -t "Your text" --no-bert --no-tfidf

# Only TF-IDF
python main.py -t "Your text" --no-bert --no-roberta
```

### Use Case 5: Get Just the Answer
When you only want the final result:

```bash
python main.py -t "Your text" --simple --quiet
```

### Use Case 6: Deep Analysis
When you want all the details:

```bash
python main.py -t "Your text" --ensemble confidence_weighted
# This is the default and gives full details
```

---

## 🐍 Python API Usage

### Basic Usage

```python
from main import FakeNewsDetector

# Initialize
detector = FakeNewsDetector()

# Analyze
text = "Your news text here"
result = detector.predict(text)

# Print formatted result
detector.print_result(result)
```

### Access Individual Values

```python
# Get specific values
prediction = result['final_prediction']     # 'FAKE' or 'REAL'
confidence = result['confidence']           # 0.0 to 1.0
risk_level = result['risk_level']           # 'HIGH', 'MEDIUM', 'LOW'
consensus = result['consensus_score']       # 0.0 to 1.0

# Get individual model predictions
bert_pred = result['individual_predictions']['BERT']
roberta_pred = result['individual_predictions']['RoBERTa']
tfidf_pred = result['individual_predictions']['TF-IDF']
```

### Custom Configuration

```python
# Custom model selection
detector = FakeNewsDetector(
    use_bert=True,
    use_roberta=True,
    use_tfidf=False,
    ensemble_method='majority_vote',
    verbose=False
)

# Analyze multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
results = detector.predict_batch(texts)

# Process results
for result in results:
    print(f"{result['final_prediction']}: {result['confidence']:.1%}")
```

### Using Individual Components

```python
# Just inference
from inference import MultiModelInference

inference = MultiModelInference()
predictions = inference.predict_all("Your text")

# Just orchestration
from orchestrator import PredictionOrchestrator, EnsembleMethod

orchestrator = PredictionOrchestrator(
    ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
)
ensemble = orchestrator.ensemble_predict(predictions)

# Just explanation
from explainer import PredictionExplainer

explainer = PredictionExplainer()
explanation = explainer.explain_prediction(
    text="Your text",
    prediction='FAKE',
    confidence=0.9,
    model_name='BERT'
)
```

---

## 📊 Understanding Output

### Terminal Output Explained

```
🚫 PREDICTION: FAKE              ← Final ensemble decision
📊 Confidence: 89.23%            ← How confident the system is
🤝 Consensus: 85.00%             ← How much models agree
⚠️  Risk Level: HIGH              ← Overall risk assessment

Individual Model Predictions:
BERT:
  Prediction: FAKE               ← BERT's decision
  Confidence: 89.50%             ← BERT's confidence
  Probabilities: REAL=10.50%, FAKE=89.50%  ← Raw probabilities

[Similar for RoBERTa and TF-IDF]

Key Findings:
  • All models agree: FAKE       ← Important observations
  • High average confidence
  • Multiple fake news indicators found
```

### Risk Levels

- **HIGH**: Take action
  - If prediction is FAKE: Very likely fake, don't share
  - If prediction is REAL: Low confidence, verify carefully

- **MEDIUM**: Be cautious
  - Moderate confidence
  - Verify from other sources

- **LOW**: Generally reliable
  - High confidence
  - But still verify important claims

- **UNKNOWN**: Cannot determine
  - System error or models disagree completely
  - Manual review needed

---

## 🎯 Tips for Best Results

### 1. Provide Complete Text
✅ Good: Full article with context
❌ Bad: Just headline or fragment

### 2. Check Consensus Score
- High consensus (>80%): More reliable
- Low consensus (<60%): Review carefully

### 3. Look at Individual Models
- All agree: Strong signal
- Disagree: Uncertain, needs human judgment

### 4. Read Explanations
The system tells you WHY it made its prediction:
- Clickbait indicators
- Emotional language
- Source credibility
- Text quality

### 5. Always Verify Important Information
This system is a tool, not a replacement for critical thinking.

---

## 🔧 Troubleshooting

### "Model not found" Error
```bash
# Check model locations
ls models/bert/final_model/
ls models/roberta/final_model/
ls models/tf_idf/

# They should contain:
# bert: model.safetensors
# roberta: model.safetensors, config.json, tokenizer files
# tf_idf: fake_news_classifier.joblib, tfidf_model.joblib
```

### Out of Memory
```bash
# Use fewer models
python main.py -t "text" --no-bert --no-roberta

# Or just use TF-IDF (smallest)
python main.py -t "text" --no-bert --no-roberta
```

### Slow Performance
```bash
# Simplify output
python main.py -t "text" --simple --no-explain

# Use quiet mode
python main.py -t "text" -q
```

### Import Errors
```bash
# Install/reinstall dependencies
pip install -r requirements.txt

# Or install individually
pip install torch transformers scikit-learn joblib numpy
```

---

## 📝 Examples

### Example 1: Social Media Post
```bash
python demo.py
# Paste: "BREAKING: Celebrity spotted doing SHOCKING thing!!!"
```

### Example 2: News Article
```bash
# Save article to file: article.txt
python main.py -f article.txt
```

### Example 3: Multiple Sources
```bash
# Create batch file
cat > sources.txt << EOF
Source 1: Article text here...
Source 2: Another article...
Source 3: Third article...
EOF

python main.py -b sources.txt
```

### Example 4: Compare Ensemble Methods
```bash
TEXT="Scientists make breakthrough discovery"

python main.py -t "$TEXT" -e majority_vote -s
python main.py -t "$TEXT" -e confidence_weighted -s
python main.py -t "$TEXT" -e adaptive -s
```

---

## 🎓 Learning Path

**Beginner**: Start here
1. Run `python demo.py` - See it in action
2. Run `python test.py` - Verify it works
3. Try `bash run.sh` - Explore features

**Intermediate**: Dive deeper
1. Use `main.py` with different options
2. Try different ensemble methods
3. Run examples: `python examples.py --example 1`

**Advanced**: Full control
1. Use Python API for integration
2. Customize model weights
3. Build your own workflows

---

## 📞 Quick Reference

```bash
# Most common commands
python demo.py                  # Interactive demo
python test.py                  # Test system
python main.py -i               # Interactive mode
python main.py -t "text"        # Analyze text
python main.py -f file.txt      # Analyze file
python main.py -b batch.txt     # Batch mode

# With options
python main.py -t "text" -s     # Simple output
python main.py -t "text" -q     # Quiet mode
python main.py -t "text" -e majority_vote  # Different ensemble

# Examples
python examples.py --all        # All examples
python examples.py --example 1  # Specific example
```

---

Remember: This is a tool to assist you, not replace critical thinking. Always verify important information from trusted sources!
