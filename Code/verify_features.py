import os
import numpy as np

def verify_dataset_features():
    target_files = [
        "positive_features_train.npy",
        "positive_features_test.npy",
        "positive_features_val.npy",
        "negative_features_train.npy",
        "negative_features_test.npy",
        "negative_features_val.npy"
    ]
    
    print("="*60)
    print("       FEATURE DATASET TENSOR INSPECTION")
    print("="*60)
    
    found_any = False
    for filename in target_files:
        candidate_paths = [
            os.path.join(".", "output", "sherlock", filename),
            os.path.join(".", "data", filename)
        ]
        
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    tensor = np.load(path)
                    print(f"[✓] {filename:30s} -> Shape: {str(tensor.shape):20s} (Total clips/frames: {tensor.shape[0]})")
                    found_any = True
                except Exception as e:
                    print(f"[!] Error loading {path}: {e}")
                    
    if not found_any:
        print("[!] No feature files (.npy) found yet. Run 'setup', 'generate', or 'inject_custom_data.py' first.")
    print("="*60)

if __name__ == "__main__":
    verify_dataset_features()
