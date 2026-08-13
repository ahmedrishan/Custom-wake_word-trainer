import os
import glob
import json
import hashlib
import numpy as np

MANIFEST_PATH = os.path.join(".", "data", "injected_manifest.json")

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_manifest(manifest_set):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(list(manifest_set), f, indent=2)

def is_validation_clip(filepath):
    """Deterministically assign 20% of files to Validation and 80% to Train based on file hash."""
    filename = os.path.basename(filepath)
    h = hashlib.md5(filename.encode('utf-8')).hexdigest()
    return (int(h, 16) % 100) < 20

def extract_and_inject():
    voice_dir = os.path.join(".", "data", "my-voice")
    all_wav_files = glob.glob(os.path.join(voice_dir, "**", "*.wav"), recursive=True)
    
    if not all_wav_files:
        print(f"[!] No custom voice recordings found in {voice_dir}")
        print("    Please run 'python record_voice.py' first to record your voice clips.")
        return
        
    manifest = load_manifest()
    new_wav_files = [f for f in all_wav_files if os.path.abspath(f) not in manifest]
    
    if not new_wav_files:
        print(f"[✓] Idempotence check: All {len(all_wav_files)} voice clips in {voice_dir} have already been injected.")
        print("    No new clips to process.")
        return

    print(f"[+] Found {len(all_wav_files)} total voice clips ({len(new_wav_files)} new clips to inject).")

    try:
        from livekit.wakeword import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
    except Exception as e:
        print(f"[*] LiveKit Wakeword AudioFeatureExtractor initialization: {e}")
        extractor = None

    train_feats_list = []
    val_feats_list = []
    successfully_processed = []

    if extractor:
        for wav_path in new_wav_files:
            try:
                feats = extractor.extract_features_from_file(wav_path)
                if feats is not None and len(feats.shape) == 3 and feats.shape[1] == 16 and feats.shape[2] == 96:
                    if is_validation_clip(wav_path):
                        val_feats_list.append(feats)
                    else:
                        train_feats_list.append(feats)
                    successfully_processed.append(os.path.abspath(wav_path))
            except Exception as ex:
                print(f"[!] Error processing {wav_path}: {ex}")

    if train_feats_list or val_feats_list:
        split_data_map = {
            "positive_features_train.npy": np.concatenate(train_feats_list, axis=0) if train_feats_list else None,
            "positive_features_val.npy": np.concatenate(val_feats_list, axis=0) if val_feats_list else None
        }

        for filename, split_data in split_data_map.items():
            if split_data is None:
                continue

            candidate_paths = [
                os.path.join(".", "output", "sherlock", filename),
                os.path.join(".", "data", filename)
            ]
            injected = False
            for path in candidate_paths:
                if os.path.exists(path):
                    existing_feats = np.load(path)
                    combined_feats = np.concatenate([existing_feats, split_data], axis=0)
                    np.save(path, combined_feats)
                    print(f"[✓] Injected into {path}: Existing {existing_feats.shape[0]} + New Custom {split_data.shape[0]} = Total {combined_feats.shape[0]} clips")
                    injected = True

            if not injected:
                fallback_path = os.path.join(".", "data", f"custom_{filename}")
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                if os.path.exists(fallback_path):
                    existing_feats = np.load(fallback_path)
                    combined_feats = np.concatenate([existing_feats, split_data], axis=0)
                else:
                    combined_feats = split_data
                np.save(fallback_path, combined_feats)
                print(f"[✓] Saved to fallback {fallback_path} ({combined_feats.shape[0]} total clips).")

        # Update manifest
        manifest.update(successfully_processed)
        save_manifest(manifest)
        print(f"[✓] Updated manifest tracking file: {MANIFEST_PATH}")
    else:
        print("[!] Feature extraction waiting for pipeline setup and dependencies.")

if __name__ == "__main__":
    extract_and_inject()
