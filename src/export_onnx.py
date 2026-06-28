import torch
from transformers import DistilBertForSequenceClassification
import os


def export_to_onnx():
    print("Loading fine-tuned baseline weights...")
    model_path = "./models/distilbert_baseline"
    output_path = "./models/moderation_model.onnx"

    # 1. Load the model directly from your local directory
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Hard-lock the model into evaluation mode (disables dropout layers)
    model.eval()

    # 2. Create dummy inputs (Batch size of 1, max sequence length of 128)
    # The ONNX exporter requires real tensor shapes to trace the computational graph.
    dummy_input_ids = torch.randint(0, 2000, (1, 128), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

    # 3. Define dynamic axes
    # By default the exported model will have the shapes of all input and output tensors set to exactly match those given in args.
    # To specify axes of tensors as dynamic (i.e. known only at run-time), set dynamic_axes to a dict.
    # We must explicitly tell the graph that batch sizes and sequence lengths will fluctuate in a production API.
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }

    print("Compiling model graph to ONNX. This may take a few seconds...")

    # 4. Execute the compilation
    torch.onnx.export(
        model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Highly stable opset for Transformer architectures
        do_constant_folding=True  # Pre-calculates static graph nodes to maximize inference speed
    )

    print(f"SUCCESS: Optimized ONNX engine saved to {output_path}")


if __name__ == "__main__":
    export_to_onnx()