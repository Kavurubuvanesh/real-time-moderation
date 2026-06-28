import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


def quantize_model():
    model_fp32 = "./models/moderation_model.onnx"
    sanitized_path = "./models/moderation_model_sanitized.onnx"
    model_quant = "./models/moderation_model_int8.onnx"

    if not os.path.exists(model_fp32):
        print(f"FATAL: Could not find {model_fp32}")
        return

    print("Step 1: Sanitizing the computational graph...")
    # 1. Load the raw ONNX graph
    model = onnx.load(model_fp32)

    # 2. Rip out the conflicting shape metadata hardcoded by PyTorch
    model.graph.ClearField("value_info")
    onnx.save(model, sanitized_path)

    print("Step 2: Executing INT8 Dynamic Quantization...")
    # 3. Quantize the clean graph
    quantize_dynamic(
        model_input=sanitized_path,
        model_output=model_quant,
        weight_type=QuantType.QUInt8
    )

    print(f"SUCCESS: Quantized engine saved to {model_quant}")

    # 4. Clean up the temporary sanitized file
    if os.path.exists(sanitized_path):
        os.remove(sanitized_path)

    # Let's prove the size reduction
    size_fp32 = os.path.getsize(model_fp32) / (1024 * 1024)
    size_int8 = os.path.getsize(model_quant) / (1024 * 1024)
    print(f"Original Size: {size_fp32:.2f} MB")
    print(f"Quantized Size: {size_int8:.2f} MB")


if __name__ == "__main__":
    quantize_model()