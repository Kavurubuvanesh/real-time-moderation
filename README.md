<div align="center">
  
# 🛡️ Real-Time Content Moderation Engine

*A sub-50ms inference microservice for real-time NLP policy violation detection.*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)

</div>

---

## ⚡ Overview
This engine provides high-speed, real-time detection of multi-label text policy violations (toxicity, severe toxicity, obscenity, threats, insults, and identity hate). 

By compiling a fine-tuned PyTorch `DistilBERT` model into a static **ONNX** computation graph and applying **dynamic INT8 quantization**, this microservice achieves extreme low-latency inference on standard CPU hardware without the need for expensive GPU hosting.

## 🏗️ Architecture & Optimization
*   **Modeling:** PyTorch, Hugging Face Transformers (`DistilBERT`)
*   **Optimization:** ONNX Runtime, INT8 Dynamic Quantization, Graph Fusion
*   **API Layer:** FastAPI, Uvicorn
*   **Performance Metrics:** 
    *   📉 **Memory Footprint:** Reduced by 75% (260MB -> 65MB)
    *   ⏱️ **Inference Speed:** <50ms per request on CPU

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone [https://github.com/Kavurubuvanesh/real-time-moderation.git](https://github.com/Kavurubuvanesh/real-time-moderation.git)
cd real-time-moderation
pip install -r requirements.txt
```

### 2. Boot the API Engine
```bash
uvicorn api.main:app --reload
```
*The API will boot and load the INT8 optimized graph into memory.*

### 3. Test the Endpoint (cURL)
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/moderate](http://127.0.0.1:8000/moderate)' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Your sample text here"
}'
```

---

## 📡 API Response Schema
The endpoint returns a clean JSON response with the latency metric and individual class probability scores:

```json
{
  "is_toxic": true,
  "latency_ms": 48.21,
  "scores": {
    "toxic": 0.985,
    "severe_toxic": 0.038,
    "obscene": 0.320,
    "threat": 0.032,
    "insult": 0.844,
    "identity_hate": 0.021
  }
}
```