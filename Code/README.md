# Custom Wake-Word Pipeline: "Sherlock"

This directory contains the custom wake-word build setup for **Sherlock** responding to:
- `"sherlock"`
- `"hey sherlock"`
- `"hi sherlock"`
- `"hello sherlock"`

Target deliverable: **`output/sherlock/sherlock.onnx`**

---

## Correct Execution Order

> [!IMPORTANT]
> The `generate` and `augment` stages construct the base synthetic positive and negative feature files. To ensure your custom voice clips **never get overwritten**, run **`inject_custom_data.py` AFTER `augment` and BEFORE `train`**.

### Step 1: Record Personal Voice Clips & Room Noise
Run the interactive voice recorder helper:
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\ahmed\miniconda3\envs\wakeword\python.exe record_voice.py
```
* **Option 1 (Voice clips)**: Record 20–30 short clips for each phrase.
* **Option 2 (Room noise)**: Record 10–15 minutes of continuous background room noise.

Clips save into `./data/my-voice/` and `./data/my-room-noise/`.

---

### Step 2: Download Dependencies, Generate Base Data & Augment Noise
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe setup -c sherlock.yaml
C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe generate -c sherlock.yaml
C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe augment -c sherlock.yaml
```

---

### Step 3: Inject Personal Voice Data & Verify Row Count
Blend your personal voice recordings into positive feature files and visually verify clip counts:
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\ahmed\miniconda3\envs\wakeword\python.exe inject_custom_data.py
C:\Users\ahmed\miniconda3\envs\wakeword\python.exe verify_features.py
```

---

### Step 4: Train Model & Export ONNX Classifier
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe train -c sherlock.yaml
C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe export -c sherlock.yaml
```

---

### Step 5: Evaluate & Sanity Check
1. **Automated Evaluation** (calculates recall against held-out personal voice clips):
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   C:\Users\ahmed\miniconda3\envs\wakeword\Scripts\livekit-wakeword.exe eval -c sherlock.yaml
   ```
2. **Model Sanity Check**:
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   C:\Users\ahmed\miniconda3\envs\wakeword\python.exe test_model.py
   ```
