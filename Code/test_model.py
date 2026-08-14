import asyncio
import json
import os
import sys
import time
import numpy as np

def get_onnx_path():
    onnx_path = os.path.join(".", "output", "sherlock", "sherlock.onnx")
    if not os.path.exists(onnx_path):
        onnx_path_alt = os.path.join(".", "sherlock.onnx")
        if os.path.exists(onnx_path_alt):
            onnx_path = onnx_path_alt
        else:
            print(f"[!] Target model file not found at {onnx_path}")
            print("    Please run the training pipeline first: 'livekit-wakeword export Code/sherlock.yaml'")
            sys.exit(1)
    return os.path.abspath(onnx_path)


async def run_live_microphone(onnx_path: str, threshold: float = 0.58):
    try:
        from livekit.wakeword import WakeWordListener, WakeWordModel
    except ImportError as e:
        print(f"[!] Unable to import livekit-wakeword: {e}")
        return

    print("\n" + "=" * 70)
    print("       SHERLOCK LIVE MICROPHONE REAL-TIME SANITY CHECK")
    print("=" * 70)
    print(f"Model Path  : {onnx_path}")
    print(f"Threshold   : {threshold:.2f} (Optimal evaluated threshold)")
    print("Target      : 'sherlock', 'hey sherlock', 'hi sherlock', 'hello sherlock'")
    print("Status      : LISTENING LIVE ON MICROPHONE... (Press Ctrl+C to stop)")
    print("=" * 70 + "\n")

    try:
        model = WakeWordModel(models=[onnx_path])
    except Exception as ex:
        print(f"[!] Failed to initialize WakeWordModel: {ex}")
        return

    detection_count = 0
    try:
        async with WakeWordListener(model, threshold=threshold, debounce=1.0) as listener:
            while True:
                detection = await listener.wait_for_detection()
                detection_count += 1
                t_str = time.strftime("%H:%M:%S")
                conf_pct = detection.confidence * 100
                print(f"[{t_str}] [✓] DETECTED #{detection_count:02d}: '{detection.name}' | Confidence: {detection.confidence:.4f} ({conf_pct:5.1f}%)")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n" + "=" * 70)
        print(f"[+] Stopped live microphone listening. Total detections logged: {detection_count}")
        print("=" * 70)
    except Exception as e:
        print(f"\n[!] Live listening error: {e}")


def run_offline_evaluation(onnx_path: str):
    import onnxruntime as ort

    print(f"[+] Loading ONNX model from: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    print("[✓] ONNX model successfully loaded with ONNX Runtime!")

    manifest_path = os.path.join(".", "data", "injected_manifest.json")
    manifest_set = set()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_list = json.load(f)
                manifest_set = set(os.path.abspath(p) for p in manifest_list)
        except Exception:
            pass

    voice_dir = os.path.join(".", "data", "my-voice")
    all_wavs = []
    if os.path.exists(voice_dir):
        for fname in sorted(os.listdir(voice_dir)):
            if fname.lower().endswith(".wav"):
                full_p = os.path.abspath(os.path.join(voice_dir, fname))
                all_wavs.append((fname, full_p))

    fresh_wavs = [(fname, p) for fname, p in all_wavs if p not in manifest_set]

    print("\n" + "=" * 70)
    print("       SHERLOCK ONNX FILE INFERENCE EVALUATION")
    print("=" * 70)
    print(f"Total voice clips in data/my-voice/ : {len(all_wavs)}")
    print(f"Original trained/val manifest clips : {len(manifest_set)}")
    print(f"Fresh brand-new test clips found   : {len(fresh_wavs)}")
    print("=" * 70)

    if not fresh_wavs:
        print("\n[*] No new un-injected voice clips found in data/my-voice/.")
        print("=" * 70)
        return

    from livekit.wakeword import WakeWordModel
    model = WakeWordModel(models=[onnx_path])
    import soundfile as sf

    print("\n[+] Evaluating Fresh Un-Injected Clips with Sliding Window Pipeline:")
    print("-" * 70)
    print(f"{'Filename':40s} | {'Score':8s} | {'Confidence':12s} | {'Detected (>0.58)'}")
    print("-" * 70)

    detected_count = 0
    scores = []

    for fname, full_p in fresh_wavs:
        try:
            audio, sr = sf.read(full_p)
            predictions = model.predict(audio)
            conf = list(predictions.values())[0] if predictions else 0.0

            scores.append(conf)
            is_detected = conf >= 0.58
            if is_detected:
                detected_count += 1

            status_str = "YES [✓]" if is_detected else "NO  [X]"
            print(f"{fname:40s} | {conf:.4f}   | {conf*100:6.2f}%      | {status_str}")
        except Exception as err:
            print(f"[!] Error processing {fname}: {err}")

    print("-" * 70)
    avg_score = np.mean(scores) if scores else 0.0
    print(f"Summary: {detected_count}/{len(fresh_wavs)} fresh clips detected ({detected_count/len(fresh_wavs)*100:.1f}% accuracy). Average confidence: {avg_score*100:.2f}%")
    print("=" * 70)


def main():
    onnx_path = get_onnx_path()
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--file", "--offline", "-f"]:
        run_offline_evaluation(onnx_path)
    else:
        try:
            asyncio.run(run_live_microphone(onnx_path, threshold=0.58))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
