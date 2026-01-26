Fake News Detection System - Architecture Document
1. Project Overview
Project: AI-powered Fake News Detection System
Core Technology: Fine-tuned BERT/RoBERTa transformer models
Objective: Detect and classify news articles as "Real" or "Fake" with explanations
Current Status: Trained models available, need production deployment

2. Problem Statement
Current Challenge:
Have successfully trained BERT and RoBERTa models on fake news datasets

Need to create a production-ready system that:

Serves model predictions reliably

Provides human-understandable explanations

Scales for multiple users/interfaces

Maintains clean separation of concerns

Technical Requirements:
Model Serving: Load and run trained transformer models efficiently

Text Processing: Clean and prepare raw news text for model input

Explanation Generation: Provide insights into why a prediction was made

Multiple Interfaces: Support CLI, API, and potential web frontend

Maintainability: Clean architecture for easy updates and debugging

3. Proposed Architecture
3.1 Core Design Principles
Separation of Concerns: Each component has a single responsibility

Single Entry Point: All external systems talk to one interface

Clean Data Flow: Predictable pipeline from input to output

Model Independence: Can switch between BERT/RoBERTa easily

Error Boundaries: Each layer handles its own errors

3.2 Architectural Layers
text
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL INTERFACES                      │
│  (CLI / API / Web Frontend / Other Applications)           │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    SERVICE LAYER                            │
│  • Orchestrator: Single public interface                    │
│  • Inference Service: Pure prediction logic                 │
│  • Explanation Service: Human-readable explanations         │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                      CORE LAYER                             │
│  • Detector: Business logic coordinator                     │
│  • Engine: Model loading and execution                      │
│  • Processors: Text preprocessing and feature extraction    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    MODEL WEIGHTS                            │
│  • BERT: Fine-tuned BERT-base-uncased model                 │
│  • RoBERTa: Fine-tuned RoBERTa-base model                   │
└─────────────────────────────────────────────────────────────┘
4. Component Breakdown
4.1 Model Weights (models/)
Purpose: Store trained model artifacts

text
models/
├── bert/                    # Fine-tuned BERT model
│   ├── config.json         # Model configuration
│   ├── model.safetensors   # Trained weights
│   └── tokenizer_config.json
└── roberta/                # Fine-tuned RoBERTa model
Key Decisions:

Use .safetensors format for security

Keep models separate from code for easy updates

Support multiple model variants

4.2 Core Layer (core/)
Purpose: Core business logic and model operations

4.2.1 Engine (core/engine/)
Responsibility: Load and execute trained models

Components:

bert_engine.py: Loads BERT model, runs inference

roberta_engine.py: Loads RoBERTa model, runs inference

Interface:

python
class ModelEngine:
    def load_model(model_path): ...
    def predict(processed_text): ...
    def get_model_info(): ...
4.2.2 Processors (core/processor/)
Responsibility: Text preprocessing and feature extraction

Components:

text_processor.py: Clean, tokenize, normalize text

feature_extractor.py: Extract linguistic features (optional)

Interface:

python
class TextProcessor:
    def preprocess(text): ...  # Returns cleaned text
    def tokenize(text): ...    # Returns tokens
4.2.3 Detector (core/detector.py)
Responsibility: Coordinate engine and processors

Optional: Can contain domain-specific business logic

Interface:

python
class FakeNewsDetector:
    def analyze(text, model_type="bert"): ...
    # Coordinates: preprocessing → model inference → result formatting
4.3 Service Layer (services/)
Purpose: Public-facing interfaces and specialized services

4.3.1 Orchestrator (services/orchestrator.py)
THE single entry point for all external systems

Responsibilities:

Validate all inputs

Coordinate service calls

Format responses consistently

Handle errors gracefully

Interface:

python
class FakeNewsOrchestrator:
    def analyze(text, detailed=False): ...
    def batch_analyze(texts): ...
    def get_system_status(): ...
4.3.2 Inference Service (services/inference.py)
Responsibility: Pure prediction logic

Characteristics:

No text preprocessing

No explanation generation

Just: text → prediction

Interface:

python
class InferenceService:
    def predict(processed_text): ...
    def get_prediction_confidence(): ...
4.3.3 Explanation Service (services/explainer.py)
Responsibility: Generate human-readable explanations

Techniques:

Attention visualization

Key phrase extraction

Confidence breakdown

Similar case references

Interface:

python
class ExplanationService:
    def explain(prediction, text): ...
    def get_key_phrases(text): ...
4.4 Supporting Components
config/: Settings, constants, environment variables

utils/: Helper functions, logging, validators

api/: REST API implementation (FastAPI)

cli/: Command-line interface

frontend/: Optional web interface

5. Data Flow
5.1 Single Prediction Flow
text
1. User Input → "Breaking news: Aliens land in New York!"
2. CLI/API → calls → services/orchestrator.py.analyze()
3. Orchestrator → validates input, logs request
4. Orchestrator → calls → services/inference.py.predict()
5. Inference Service → calls → core/detector.py.analyze()
6. Detector → calls → core/processor/text_processor.py.preprocess()
7. Detector → calls → core/engine/bert_engine.py.predict()
8. BERT Engine → loads model from models/bert/
9. BERT Engine → runs inference, returns probabilities
10. Detector → formats results
11. Inference Service → returns prediction
12. Orchestrator → calls → services/explainer.py.explain()
13. Orchestrator → formats final response
14. CLI/API → presents result to user
5.2 Response Format
json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 0.842,
  "probabilities": {
    "fake": 0.842,
    "real": 0.158
  },
  "explanation": {
    "key_phrases": ["aliens land", "breaking news"],
    "confidence_level": "High",
    "warning_signs": ["Sensational language", "No sources cited"]
  },
  "metadata": {
    "model_used": "bert",
    "processing_time_ms": 125,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
6. Why This Architecture?
6.1 Benefits
Clean Separation of Concerns

Model weights isolated from code

Business logic separate from interfaces

Services independent and testable

Scalability

Can add new models without changing interfaces

Services can be deployed independently

Easy to add new interfaces (CLI, API, Web)

Maintainability

Clear dependencies between components

Easy to debug (know exactly where issues occur)

Simple to update models or processing logic

Flexibility

Switch between BERT/RoBERTa with one parameter

Add new explanation methods easily

Support batch processing and streaming

Production Ready

Proper error handling at each layer

Logging and monitoring ready

Configurable through settings

6.2 Comparison with Alternatives
Approach	Pros	Cons	Why We Chose Our Design
Monolithic	Simple to start	Hard to maintain, test, scale	Need clean separation for production
Service-Heavy	Very modular	Over-engineering for our needs	Balanced: enough services without complexity
Model-Centric	Focused on ML	Poor separation from business logic	Need both ML and business logic
Our Design	Balanced, clean, scalable	Slight learning curve	Best trade-off for our requirements
7. Implementation Roadmap
Phase 1: Core Foundation (Current)
Train BERT/RoBERTa models

Define architecture

Implement core/engine/ (model loading)

Implement core/processor/ (text processing)

Phase 2: Service Layer
Implement services/orchestrator.py

Implement services/inference.py

Implement services/explainer.py

Set up configuration system

Phase 3: Interfaces
CLI interface (cli/)

REST API (api/)

Basic frontend (optional)

Phase 4: Productionization
Error handling and logging

Performance optimization

Monitoring and metrics

Documentation

8. Technical Decisions Rationale
Why separate models/ from core/?

Models are large binary files (100s of MB)

Code and data should be separate

Easy to update models without touching code

Why have both engine/ and inference.py?

engine/: Technical model execution (load, run GPU/CPU)

inference.py: Service-level prediction logic (caching, validation)

Separation allows different optimization strategies

Why orchestrator.py as single entry point?

Consistent error handling

Single place for logging/monitoring

Easy to add new features (rate limiting, authentication)

Why support multiple models (BERT/RoBERTa)?

Different models have different strengths

Allows A/B testing

Provides fallback options

9. Success Metrics
Performance:

Inference time < 500ms per request

99% service availability

Support 100+ concurrent users

Accuracy:

Maintain trained model accuracy (75-85% based on tests)

Low false positive/negative rates

Usability:

Clear explanations for predictions

Easy integration via API/CLI

Comprehensive documentation

Maintainability:

Clear separation for debugging

Easy to update models

Test coverage > 80%

10. Next Steps
Immediate:

Finalize component interfaces

Implement core/engine/bert_engine.py

Set up basic text processor

Short-term:

Build service layer integration

Create CLI interface

Add basic explanation service

Long-term:

Add advanced features (attention visualization)

Support more models

Deploy as microservice

Document Version: 1.0
Last Updated: January 2024
Status: Approved for Implementation
