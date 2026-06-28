# Real-Time Content Moderation Engine

A high-performance, low-latency microservice for real-time text moderation. This engine detects multi-label policy violations (toxicity, severe toxicity, obscenity, threats, insults, and identity hate) in textual data streams.

By compiling a fine-tuned PyTorch `DistilBERT` model into a static **ONNX** computation graph and applying **dynamic INT8 quantization**, this engine achieves sub-50ms inference latency on standard CPU hardware.

## Architecture & Optimization
*   **Modeling:** PyTorch, Hugging Face Transformers (`DistilBERT`)
*   **Optimization:** ONNX Runtime, INT8 Dynamic Quantization, Graph Fusion
*   **API Layer:** FastAPI, Uvicorn
*   **Performance:** Reduced model memory footprint by 75% (260MB -> 65MB) and achieved <50ms CPU inference speed by stripping Python framework overhead.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Boot the API Engine
```bash
uvicorn api.main:app --reload
```

### 3. Test the Endpoint
Send a POST request to `http://127.0.0.1:8000/moderate`:
```json
{
  "text": "Your sample text here"
}
```

## Response Schema
```json
{
  "is_toxic": true,
  "latency_ms": 48.21,
  "scores": {
    "toxic": 0.98,
    "severe_toxic": 0.03,
    "obscene": 0.32,
    "threat": 0.01,
    "insult": 0.84,
    "identity_hate": 0.02
  }
}
```