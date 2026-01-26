# Text Detection System

A modular, production-oriented architecture for transformer-based text detection and analysis. This system emphasizes clean separation of concerns, maintainability, and extensibility.

> **Note:** This repository provides the inference and orchestration framework. Model training is handled separately—see [Model Training](#model-training) below.

---

## Design Philosophy

This project is built around four core principles:

1. **Separation of Concerns** – Clear boundaries between preprocessing, inference, and application logic
2. **Model Agnosticism** – Interface-level abstractions allow model swapping without system rewrites
3. **Data Flow Transparency** – Explicit pipelines from raw input to final predictions
4. **Research & Production Ready** – Suitable for experimentation while maintaining production-grade structure

Unlike monolithic scripts, this architecture allows models, preprocessing strategies, and inference pipelines to evolve independently.

---

## Architecture Overview

### System Components

**Core Layer** (`core/`)  
The detection engine and its dependencies:
- `detector.py` – Central detection orchestrator
- `engine/` – Model invocation interfaces (BERT, custom models)
- `processor/` – Text normalization and feature extraction

**Services Layer** (`services/`)  
High-level application workflows:
- `inference.py` – End-to-end prediction pipeline
- `explainer.py` – Model interpretability and output analysis
- `orchestrator.py` – Multi-step workflow coordination

**Configuration & Utilities**  
- `config/` – Environment settings, hyperparameters, model paths
- `utils/` – Shared helpers, logging, validation

**Entry Point**  
- `main.py` – CLI and programmatic interface

### Data Flow

```
Raw Text Input
    ↓
Text Processor (normalization, tokenization)
    ↓
Feature Extractor (embeddings, metadata)
    ↓
Model Engine (BERT/transformer inference)
    ↓
Detector (classification, scoring)
    ↓
Services (interpretation, orchestration)
    ↓
Final Output (predictions, explanations)
```

---

## Model Training

This repository **does not include model training code or pretrained weights**. Model development—including fine-tuning for low-resource scenarios—is maintained separately:

### SPST-BERT Training Repository
**[github.com/premananda-cloud/Bert_training_via_SPST](https://github.com/premananda-cloud/Bert_training_via_SPST)**

This external repository implements **Sequential/Single-Phase Single-Task (SPST)** training optimized for:
- Limited datasets and compute resources
- Reproducible research workflows
- Controlled fine-tuning experiments

**Integration:** Once trained, model artifacts (checkpoints, configs) are loaded into this system via `core/engine/call_bert.py`.

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <your-project>

# Install dependencies
pip install -r requirements.txt

# Configure model paths
# Edit config/settings.yaml to point to your trained model
```

---

## Usage

### Basic Inference

```python
from services.inference import InferencePipeline

pipeline = InferencePipeline(model_path="path/to/model")
result = pipeline.predict("Sample text for detection")
print(result)
```

### With Explanations

```python
from services.explainer import Explainer

explainer = Explainer(pipeline)
explanation = explainer.interpret(result)
```

### Command Line

```bash
python main.py --input sample.txt --output predictions.json
```

---

## Configuration

All system behavior is controlled through `config/`:

- **Model Settings** – Path to trained model, tokenizer configuration
- **Processing Parameters** – Max sequence length, preprocessing rules
- **Inference Options** – Batch size, confidence thresholds
- **Logging & Output** – Verbosity, export formats

Example `config/settings.yaml`:

```yaml
model:
  path: "models/spst-bert-detector"
  device: "cuda"
  
preprocessing:
  max_length: 512
  lowercase: true
  
inference:
  batch_size: 32
  threshold: 0.7
```

---

## Extending the System

### Adding a New Model

1. Implement model interface in `core/engine/`
2. Register in detector configuration
3. Update `services/inference.py` if needed

### Custom Preprocessing

1. Extend `processor/text_processor.py`
2. Add feature extractors in `processor/feature_extracter.py`
3. Update pipeline configuration

### New Service Workflows

Create new modules in `services/` following existing patterns for inference and explanation.

---

## Intended Use Cases

- **Research Prototypes** – Rapid experimentation with new models or architectures
- **Evaluation Frameworks** – Systematic testing of transformer-based detectors
- **Integration Testing** – Validating externally trained models in realistic workflows
- **Production Foundation** – Starting point for deployment (requires additional hardening)

---

## Important Notes

- This system intentionally separates inference from training to maintain clean abstractions
- Users are responsible for validating models before deployment
- No pretrained weights are included—bring your own trained models
- All architectural decisions prioritize clarity and reproducibility over convenience

---

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- NumPy, pandas

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions are welcome. Please:
1. Follow existing architectural patterns
2. Add tests for new functionality
3. Update documentation accordingly

---

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

---

**Note:** This is a research and development framework. Ensure proper validation and testing before deploying in production environments.
