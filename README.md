# Fake News Detection System

A comprehensive multi-model fake news detection system using BERT, RoBERTa, and TF-IDF models with ensemble predictions and explainability.

## 🌟 Features

- **Multi-Model Architecture**: Combines BERT, RoBERTa, and TF-IDF models for robust predictions
- **Ensemble Methods**: Multiple ensemble strategies including confidence-weighted, majority vote, and adaptive methods
- **Explainability**: Detailed explanations for predictions with text analysis and indicator detection
- **Flexible Interface**: CLI, batch processing, and interactive modes
- **Risk Assessment**: Automatic risk level calculation (HIGH/MEDIUM/LOW)
- **Comprehensive Analysis**: Text features, sentiment analysis, and credibility indicators

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full dependencies

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt

# Ensure model files are in place
# models/bert/final_model/
# models/roberta/final_model/
# models/tf_idf/fake_news_classifier.joblib
# models/tf_idf/tfidf_model.joblib
```

## 📁 Project Structure

```
├── main.py                          # Main entry point with CLI
├── services/
│   ├── inference.py                 # Model inference for all three models
│   ├── orchestrator.py              # Ensemble prediction logic
│   └── explainer.py                 # Explanation and interpretability
├── models/
│   ├── bert/final_model/           # BERT model files
│   ├── roberta/final_model/        # RoBERTa model files
│   └── tf_idf/                     # TF-IDF model and vectorizer
└── requirements.txt                 # Python dependencies
```

## 💻 Usage

### Quick Start (No Arguments Required!)

The easiest way to get started:

```bash
# Run the demo with sample texts
python demo.py --demo

# Or start interactive mode (default)
python demo.py

# Or use the menu-driven interface
bash run.sh
```

### Command Line Interface (Full Features)

#### Analyze a single text:
```bash
python main.py --text "Scientists discover groundbreaking cure for cancer"
```

#### Analyze text from file:
```bash
python main.py --file article.txt
```

#### Batch processing:
```bash
# Create a file with one text per line
python main.py --batch news_articles.txt
```

#### Interactive mode:
```bash
python main.py --interactive
```

### Advanced Options

#### Choose specific models:
```bash
# Use only BERT and RoBERTa
python main.py --text "News text..." --no-tfidf

# Use only TF-IDF
python main.py --text "News text..." --no-bert --no-roberta
```

#### Select ensemble method:
```bash
python main.py --text "News..." --ensemble majority_vote
# Available: majority_vote, weighted_average, unanimous, confidence_weighted, adaptive
```

#### Output options:
```bash
# Simple output
python main.py --text "News..." --simple

# No explanations
python main.py --text "News..." --no-explain

# Quiet mode
python main.py --text "News..." --quiet
```

### Python API

```python
from main import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector(
    use_bert=True,
    use_roberta=True,
    use_tfidf=True,
    ensemble_method='confidence_weighted'
)

# Analyze text
text = "Breaking news: Scientists discover miracle cure!"
result = detector.predict(text, explain=True)

# Print formatted result
detector.print_result(result, detailed=True)

# Access prediction data
print(f"Prediction: {result['final_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Using Individual Components

#### Inference Service
```python
from services.inference import MultiModelInference

# Initialize inference
inference = MultiModelInference(
    use_bert=True,
    use_roberta=True,
    use_tfidf=True
)

# Get predictions from all models
predictions = inference.predict_all(text)

# Access individual model predictions
print(predictions['BERT'])
print(predictions['RoBERTa'])
print(predictions['TF-IDF'])
```

#### Orchestrator Service
```python
from services.orchestrator import PredictionOrchestrator, EnsembleMethod

# Initialize orchestrator
orchestrator = PredictionOrchestrator(
    ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
)

# Get ensemble prediction
ensemble_result = orchestrator.ensemble_predict(predictions)

# Print interpretation
interpretation = orchestrator.interpret_results(ensemble_result)
print(interpretation)
```

#### Explainer Service
```python
from services.explainer import PredictionExplainer

# Initialize explainer
explainer = PredictionExplainer()

# Get explanation for a prediction
explanation = explainer.explain_prediction(
    text=text,
    prediction='FAKE',
    confidence=0.92,
    model_name='BERT'
)

# Get ensemble explanation
ensemble_explanation = explainer.explain_ensemble(
    text=text,
    individual_predictions=predictions,
    ensemble_result=ensemble_result
)
```

## 🎯 Ensemble Methods

### 1. Confidence Weighted (Default)
Weights predictions by their confidence scores. Best for balanced results.

### 2. Majority Vote
Simple voting system. Each model gets one vote.

### 3. Weighted Average
Uses predefined model weights. Customize with `model_weights` parameter.

### 4. Unanimous
Requires all models to agree. Returns UNCERTAIN if they disagree.

### 5. Adaptive
Chooses strategy based on model agreement level.

## 📊 Output Format

```
======================================================================
FAKE NEWS DETECTION RESULT
======================================================================

🚫 PREDICTION: FAKE
📊 Confidence: 89.23%
🤝 Consensus: 85.00%
⚠️  Risk Level: HIGH

----------------------------------------------------------------------
INDIVIDUAL MODEL PREDICTIONS
----------------------------------------------------------------------

BERT:
  Prediction: FAKE
  Confidence: 89.50%
  Probabilities: REAL=10.50%, FAKE=89.50%

RoBERTa:
  Prediction: FAKE
  Confidence: 92.00%
  Probabilities: REAL=8.00%, FAKE=92.00%

TF-IDF:
  Prediction: REAL
  Confidence: 65.00%
  Probabilities: REAL=65.00%, FAKE=35.00%

----------------------------------------------------------------------
KEY FINDINGS
----------------------------------------------------------------------
  • ⚠️  Models disagree: 2 predict FAKE, 1 predict REAL
  • ✓ High average confidence: 82.2%
  • ⚠️  Multiple fake news indicators found (5 total)

----------------------------------------------------------------------
RECOMMENDATION
----------------------------------------------------------------------
⛔ This content is likely FAKE NEWS. Do NOT share or trust this information.
======================================================================
```

## 🔍 Text Analysis Features

The explainer analyzes:
- **Text Features**: Word count, sentence structure, capitalization patterns
- **Fake Indicators**: Clickbait phrases, emotional language, unreliable sources
- **Credibility Indicators**: Source attribution, balanced language, specific details
- **Readability**: Complexity score and reading level
- **Sentiment**: Positive/negative emotional content

## 🎨 Risk Levels

- **HIGH**: Strong indication of fake news or low confidence in authenticity
- **MEDIUM**: Moderate confidence, verification recommended
- **LOW**: High confidence in assessment
- **UNKNOWN**: Unable to determine (errors or insufficient data)

## 🛠️ Customization

### Adjust Model Weights
```python
custom_weights = {
    'BERT': 0.3,
    'RoBERTa': 0.5,
    'TF-IDF': 0.2
}

orchestrator = PredictionOrchestrator(
    ensemble_method=EnsembleMethod.WEIGHTED_AVERAGE,
    model_weights=custom_weights
)
```

### Add Custom Indicators
```python
explainer = PredictionExplainer()

# Add custom fake news indicators
explainer.fake_indicators['custom'] = [
    'phrase1', 'phrase2', 'phrase3'
]

# Add custom credibility indicators
explainer.credibility_indicators['custom'] = [
    'verified by', 'confirmed by', 'according to experts'
]
```

## 📈 Performance Tips

1. **GPU Acceleration**: Models automatically use GPU if available
2. **Batch Processing**: Use batch mode for multiple texts to improve efficiency
3. **Model Selection**: Disable unused models to reduce memory usage
4. **Simple Output**: Use `--simple` flag for faster processing

## 🐛 Troubleshooting

### Model Loading Errors
```bash
# Ensure model files are in correct locations
ls models/bert/final_model/
ls models/roberta/final_model/
ls models/tf_idf/
```

### Memory Issues
```bash
# Use fewer models
python main.py --text "..." --no-bert

# Or process in smaller batches
```

### CUDA Errors
```bash
# Force CPU mode if GPU issues occur
export CUDA_VISIBLE_DEVICES=""
```

## 📝 Examples

### Example 1: Quick Check
```bash
python main.py -t "Scientists announce breakthrough in renewable energy" -s
```

### Example 2: Detailed Analysis
```bash
python main.py -f suspicious_article.txt -e confidence_weighted
```

### Example 3: Batch Analysis
```bash
# Create file: news_batch.txt with one article per line
python main.py -b news_batch.txt --no-explain
```

### Example 4: Python Script
```python
from main import FakeNewsDetector

detector = FakeNewsDetector(verbose=False)

articles = [
    "Breaking: Aliens land in Times Square!",
    "Study shows effectiveness of new vaccine",
    "You won't believe this shocking secret!"
]

for article in articles:
    result = detector.predict(article, explain=False)
    print(f"{result['final_prediction']}: {article[:50]}...")
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ensemble methods
- More sophisticated text analysis
- Model fine-tuning utilities
- Web interface
- API endpoint

## 📄 License

See LICENSE.md for details.

## 🙏 Acknowledgments

- BERT: Google Research
- RoBERTa: Facebook AI
- Transformers library: Hugging Face

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This system is for educational and research purposes. Always verify important information from multiple trusted sources.
