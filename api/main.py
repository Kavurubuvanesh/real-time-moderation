from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
import onnxruntime as ort
import numpy as np
import time
import os

app = FastAPI(title="Real-Time Content Moderation Engine", version="1.0.0")

# --- GLOBAL STATE ---
# We load the heavy assets into memory once on startup, NOT on every request.
tokenizer = None
ort_session = None


class ModerationRequest(BaseModel):
    text: str


class ModerationResponse(BaseModel):
    is_toxic: bool
    latency_ms: float
    scores: dict


# The 6 classes our model was trained to predict
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


@app.on_event("startup")
async def load_assets():
    global tokenizer, ort_session
    print("BOOTING ENGINE: Loading tokenizer and ONNX graph into memory...")

    tokenizer_path = "./models/distilbert_baseline"
    onnx_model_path = "./models/moderation_model_int8.onnx"

    if not os.path.exists(onnx_model_path):
        raise RuntimeError(f"FATAL: ONNX model not found at {onnx_model_path}")

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    # --- LOCAL OPTIMIZATION ---
    sess_options = ort.SessionOptions()

    # 1. Keep the aggressive math/graph fusions
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 2. REMOVE the thread limits. We let ONNX dynamically consume your available CPU cores.
    # (By not setting intra/inter threads, ONNX defaults to utilizing all logical cores)

    ort_session = ort.InferenceSession(
        onnx_model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    print("ENGINE READY: Inference session active.")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_text(request: ModerationRequest):
    start_time = time.time()

    # 1. Tokenize the incoming string (Max length must match our ONNX export shape: 128)
    inputs = tokenizer(
        request.text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )

    # ONNX requires numpy arrays, not PyTorch tensors
    input_ids = np.array([inputs['input_ids']], dtype=np.int64)
    attention_mask = np.array([inputs['attention_mask']], dtype=np.int64)

    # 2. Run the high-speed inference
    ort_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Execute graph
    logits = ort_session.run(None, ort_inputs)[0]

    # 3. Process outputs
    probs = sigmoid(logits[0])

    scores = {label: float(prob) for label, prob in zip(LABELS, probs)}

    # If any category scores above 0.5, flag the text
    is_toxic = any(prob > 0.5 for prob in probs)

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    return {
        "is_toxic": is_toxic,
        "latency_ms": round(latency_ms, 2),
        "scores": scores
    }