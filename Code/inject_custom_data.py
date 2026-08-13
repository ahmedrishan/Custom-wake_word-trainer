import os
import glob
import numpy as np

def extract_and_inject():
    voice_dir = os.path.join(".", "data", "my-voice")
    wav_files = glob.glob(os.path.join(voice_dir, "*.wav"))
    
    if not wav_files:
        print(f"[!] No custom voice recordings found in {voice_dir}")
        print("    Please run 'python record_voice.py' first to record your voice clips.")
        return
        
    print(f"[+] Found {len(wav_files)} custom voice recording clips.")
    
    try:
        from livekit.wakeword import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
    except Exception as e:
        print(f"[*] LiveKit Wakeword AudioFeatureExtractor initialization: {e}")
        extractor = None

    features_list = []
    
    if extractor:
        for wav_path in wav_files:
            try:
                feats = extractor.extract_features_from_file(wav_path)
                # Ensure feature shape is (N, 16, 96)
                if feats is not None and len(feats.shape) == 3 and feats.shape[1] == 16 and feats.shape[2] == 96:
                    features_list.append(feats)
            except Exception as ex:
                print(f"[!] Error processing {wav_path}: {ex}")
                
    if features_list:
        custom_features = np.concatenate(features_list, axis=0)
        print(f"[+] Extracted custom features tensor shape: {custom_features.shape}")
        
        # Target feature files generated during `livekit-wakeword setup`
        target_npy_paths = [
            os.path.join(".", "output", "sherlock", "positive_features_train.npy"),
            os.path.join(".", "data", "positive_features_train.npy")
        ]
        
        injected = False
        for npy_path in target_npy_paths:
            if os.path.exists(npy_path):
                existing_feats = np.load(npy_path)
                print(f"[+] Existing features in {npy_path}: {existing_feats.shape}")
                combined_feats = np.concatenate([existing_feats, custom_features], axis=0)
                np.save(npy_path, combined_feats)
                print(f"[✓] Successfully injected custom features! New shape: {combined_feats.shape} -> Saved to {npy_path}")
                injected = True
                
        if not injected:
            save_path = os.path.join(".", "data", "custom_positive_features.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, custom_features)
            print(f"[✓] Saved custom positive features to {save_path}.")
    else:
        print("[!] Feature extraction waiting for pipeline setup and dependencies.")

if __name__ == "__main__":
    extract_and_inject()
