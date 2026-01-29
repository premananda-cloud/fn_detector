# Quick Start Guide - Fake News Detection System

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Models
Ensure your models are in the correct locations:
```
models/
├── bert/final_model/model.safetensors
├── roberta/final_model/model.safetensors
└── tf_idf/
    ├── fake_news_classifier.joblib
    └── tfidf_model.joblib
```

### Step 3: Run Your First Detection
```bash
python main.py --text "Breaking news: Scientists discover miracle cure!"
```

That's it! You should see a detailed analysis of the text.

## 📝 Common Use Cases

### 1. Analyze a News Article
```bash
# From text
python main.py -t "Your news text here..."

# From file
python main.py -f article.txt
```

### 2. Check Multiple Articles
```bash
# Create a file with one article per line
python main.py -b articles.txt
```

### 3. Interactive Mode (for testing)
```bash
python main.py -i
```

### 4. Quick Check (simplified output)
```bash
python main.py -t "News text..." -s
```

## 🐍 Python API Quick Example

```python
from main import FakeNewsDetector

# Initialize
detector = FakeNewsDetector()

# Analyze
result = detector.predict("Your news text here")

# Print result
detector.print_result(result)

# Access data
print(f"Prediction: {result['final_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## ⚙️ Configuration Options

### Use Specific Models
```bash
# Only BERT
python main.py -t "text" --no-roberta --no-tfidf

# Only RoBERTa and TF-IDF
python main.py -t "text" --no-bert
```

### Change Ensemble Method
```bash
python main.py -t "text" --ensemble majority_vote
# Options: majority_vote, weighted_average, confidence_weighted, unanimous, adaptive
```

### Adjust Output Detail
```bash
# Simple output
python main.py -t "text" --simple

# Without explanations
python main.py -t "text" --no-explain

# Quiet mode (minimal)
python main.py -t "text" --quiet
```

## 📊 Understanding Output

```
🚫 PREDICTION: FAKE        # Final ensemble prediction
📊 Confidence: 89.23%      # How confident the system is
🤝 Consensus: 85.00%       # How much models agree
⚠️  Risk Level: HIGH        # Overall risk assessment
```

### Risk Levels Explained
- **HIGH**: Strong indication (take action)
- **MEDIUM**: Moderate confidence (verify)
- **LOW**: Low concern (likely accurate)
- **UNKNOWN**: System couldn't determine

## 🎯 Tips for Best Results

1. **Provide complete text**: More context = better predictions
2. **Check all three models**: Each model has different strengths
3. **Look at consensus**: High consensus = more reliable
4. **Read explanations**: Understand *why* the prediction was made
5. **Verify important claims**: Always cross-check critical information

## 🔍 What the System Checks

- **Clickbait phrases**: "You won't believe", "shocking", etc.
- **Emotional language**: Excessive caps, exclamation marks
- **Source reliability**: Anonymous sources, vague attributions
- **Text quality**: Grammar, structure, complexity
- **Credibility indicators**: Citations, data, balanced language

## 💡 Example Workflow

```bash
# 1. Quick check
python main.py -t "Your news text" -s

# 2. If suspicious, get detailed analysis
python main.py -t "Your news text" -e confidence_weighted

# 3. Batch check similar articles
python main.py -b related_articles.txt

# 4. Review individual model predictions
python main.py -t "Your news text"  # Full output
```

## 🆘 Common Issues

### "Model not found" error
```bash
# Check if model files exist
ls models/bert/final_model/
ls models/roberta/final_model/
ls models/tf_idf/
```

### Out of memory
```bash
# Use fewer models
python main.py -t "text" --no-bert

# Or reduce batch size
```

### Slow processing
```bash
# GPU not detected - check CUDA installation
nvidia-smi

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

## 📚 Learn More

- Run examples: `python examples.py --all`
- Read full README: `README.md`
- Check API docs: See inline documentation in source files

## 🎓 Next Steps

1. Try the examples: `python examples.py --example 1`
2. Experiment with different ensemble methods
3. Test on real news articles
4. Customize for your specific needs
5. Integrate into your workflow

## ⚡ Performance Tips

- GPU: ~2-3 seconds per article
- CPU: ~5-10 seconds per article
- Batch mode: More efficient for multiple texts
- Simple mode: Faster, less detailed

## 🎨 Customization Quick Reference

```python
# Custom model weights
orchestrator = PredictionOrchestrator(
    model_weights={'BERT': 0.4, 'RoBERTa': 0.4, 'TF-IDF': 0.2}
)

# Custom ensemble method
detector = FakeNewsDetector(
    ensemble_method='adaptive'
)

# Selective models
detector = FakeNewsDetector(
    use_bert=True,
    use_roberta=False,
    use_tfidf=True
)
```

## 📞 Get Help

- Examples not working? Check model files are present
- Want more features? See `examples.py` for advanced usage
- Need API integration? Check Python API section in README
- Found a bug? Check GitHub issues

---

**Ready to go?** Start with: `python main.py -i` for interactive mode!
