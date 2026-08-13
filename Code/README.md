# Custom Wake-Word Pipeline: "Sherlock"

This directory contains the custom wake-word build setup for **Sherlock** responding to:
- `"sherlock"`
- `"hey sherlock"`
- `"hi sherlock"`
- `"hello sherlock"`

Target deliverable: **`output/sherlock/sherlock.onnx`**

---

## Workflow Steps

### Step 1: Record Personal Voice Clips & Room Noise
Run the interactive voice recorder helper:
```bash
C:\Users\ahmed\miniconda3\envs\wakeword\python.exe record_voice.py
```
* **Option 1 (Voice clips)**: Record 20–30 short clips for each phrase.
* **Option 2 (Room noise)**: Record 10–15 minutes of continuous background room noise.

Clips will automatically save into `./data/my-voice/` and `./data/my-room-noise/`.

---

### Step 2: Run LiveKit Pipeline Setup
Download base datasets, noise profiles, and pre-computed synthetic embeddings:
```bash
livekit-wakeword setup sherlock.yaml
```

---

### Step 3: Inject Personal Voice Data
Blend your personal voice recordings into the positive training features:
```bash
C:\Users\ahmed\miniconda3\envs\wakeword\python.exe inject_custom_data.py
```

---

### Step 4: Run Training & ONNX Export
Generate synthetic TTS data, run data augmentations (with room noise), train model, and export `sherlock.onnx`:
```bash
livekit-wakeword run sherlock.yaml
```

---

### Step 5: Evaluate & Sanity Check
1. **Automated Evaluation**:
   ```bash
   livekit-wakeword eval sherlock.yaml
   ```
2. **Model Sanity Check**:
   ```bash
   C:\Users\ahmed\miniconda3\envs\wakeword\python.exe test_model.py
   ```
