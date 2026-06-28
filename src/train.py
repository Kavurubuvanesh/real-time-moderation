import torch
import time
import numpy as np
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import precision_score
from dataset import get_data_loaders


def train_model():
    # 1. Hardware Routing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"IGNITION: Routing compute to {device}...")

    # 2. Ingest Data
    train_loader, val_loader, tokenizer = get_data_loaders(batch_size=16)

    # 3. Initialize Model Structure
    print("Loading DistilBERT backbone...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6,  # 6 specific toxicity classes
        problem_type="multi_label_classification"
    )
    model.to(device)

    # Multi-label loss requires BCE (Binary Cross Entropy) with Logits
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epochs = 1  # Sprint 1: We just need a functional baseline, not perfection yet

    print(">>> INITIATING TRAINING LOOP <<<")
    for epoch in range(epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Shovel tensors to GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- VALIDATION & LATENCY BENCHMARKING ---
        print("\nBenchmarking baseline latency and precision...")
        model.eval()
        val_preds, val_labels = [], []

        start_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)

                # Convert raw logits to probabilities, then threshold at 0.5 for binary classification per label
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > 0.5).float()

                val_preds.append(preds.cpu().numpy())
                val_labels.append(batch['labels'].numpy())

        end_time = time.time()

        # Calculate inference speed
        total_samples = len(val_loader.dataset)
        latency_ms = ((end_time - start_time) / total_samples) * 1000

        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)

        # We target Micro Precision because we want to minimize false positives in moderation
        precision = precision_score(val_labels, val_preds, average='micro')

        print(f"\n--- SPRINT 1 BASELINE RESULTS ---")
        print(f"Validation Precision: {precision:.4f} (Targeting > 0.91)")
        print(f"Raw PyTorch Latency:  {latency_ms:.2f} ms/sample (Targeting < 50ms)")

    print("\nSaving PyTorch baseline weights...")
    model.save_pretrained('./models/distilbert_baseline')
    tokenizer.save_pretrained('./models/distilbert_baseline')
    print("DONE. Ready for ONNX compilation.")


if __name__ == '__main__':
    train_model()