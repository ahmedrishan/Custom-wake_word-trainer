import os
import sys
import time
import numpy as np

def test_onnx_model():
    onnx_path = os.path.join(".", "output", "sherlock", "sherlock.onnx")
    if not os.path.exists(onnx_path):
        onnx_path_alt = os.path.join(".", "sherlock.onnx")
        if os.path.exists(onnx_path_alt):
            onnx_path = onnx_path_alt
        else:
            print(f"[!] Target model file not found at {onnx_path}")
            print("    Please run the training pipeline first: 'livekit-wakeword run sherlock.yaml'")
            return

    print(f"[+] Loading ONNX model from: {onnx_path}")
    
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        print("[✓] ONNX model successfully loaded with ONNX Runtime!")
        
        # Print input & output specs
        for i in session.get_inputs():
            print(f"    Input Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        for o in session.get_outputs():
            print(f"    Output Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
            
    except ImportError:
        print("[!] onnxruntime is not installed. Installing...")
        os.system(f"{sys.executable} -m pip install onnxruntime")
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        
    print("\n[+] Model Sanity Check complete.")

if __name__ == "__main__":
    test_onnx_model()
