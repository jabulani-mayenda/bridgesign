import os
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx
from gesture_feature_extractor import GESTURE_WINDOW
from lstm_gesture_model import GestureLSTM

def convert():
    model_path = os.path.join("models", "gesture_lstm.pt")
    meta_path = os.path.join("models", "gesture_lstm_meta.json")
    out_path = os.path.join("models", "gesture_lstm.onnx")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"Missing PyTorch model or metadata. Cannot convert.")
        return

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    input_size = checkpoint.get("input_size", 42)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 2)
    num_classes = len(checkpoint["classes"])

    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export using the same sequence length the runtime feeds by default.
    # Dynamic axes keep the exported graph flexible for shorter/longer windows.
    seq_len = GESTURE_WINDOW
    dummy_input = torch.randn(1, seq_len, input_size)

    print(f"Exporting model to {out_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        out_path, 
        export_params=True, 
        dynamo=False,
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'},
        },
    )
    print("Export complete.")

    # Validate
    print("Validating ONNX model...")
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    print("Testing inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(out_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Compare ONNX Runtime and PyTorch results
    torch_out = model(dummy_input)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("ONNX Runtime and PyTorch outputs match for the default window.")

    print("Testing a second sequence length to verify dynamic axes...")
    alt_len = max(4, GESTURE_WINDOW - 3)
    alt_input = torch.randn(1, alt_len, input_size)
    alt_inputs = {ort_session.get_inputs()[0].name: to_numpy(alt_input)}
    alt_outs = ort_session.run(None, alt_inputs)
    alt_torch_out = model(alt_input)
    np.testing.assert_allclose(to_numpy(alt_torch_out), alt_outs[0], rtol=1e-03, atol=1e-05)
    print("Dynamic sequence length validation passed. Conversion successful.")

if __name__ == "__main__":
    convert()
