# AGENTS.md - Custom Wake-Word Training Pipeline ("Sherlock")

## System & Pipeline Constraints
- **Pipeline Type**: LiveKit Wake-Word Training Pipeline (`livekit-wakeword`).
- **Target Phrasings**: `"sherlock"`, `"hey sherlock"`, `"hi sherlock"`, `"hello sherlock"`.
- **Target Deliverable**: `sherlock.onnx` classifier exported to `./Code/output/sherlock/sherlock.onnx`.
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, CUDA 12.4 supported).
- **Python Environment**: Conda environment `wakeword` at `C:\Users\ahmed\miniconda3\envs\wakeword\python.exe`.

## Execution Working Directory Rule
> [!IMPORTANT]
> All terminal commands and pipeline scripts for this project should be run from `d:\Work\Sherlock\Wake-Word\Code` or `d:\Work\Sherlock\Wake-Word`.

## Directory Layout
```
d:\Work\Sherlock\Wake-Word\
├── AGENTS.md
├── wake-word-plan.pdf
├── Code/
│   ├── sherlock.yaml
│   ├── record_voice.py
│   ├── inject_custom_data.py
│   ├── verify_features.py
│   ├── test_model.py
│   └── README.md
├── data/
│   ├── my-voice/          # Personal recorded phrase clips (.wav)
│   ├── my-room-noise/     # Ambient room noise recording (.wav)
│   └── backgrounds/       # Background noise profiles
└── output/                # Trained model output directory (sherlock.onnx, metrics.json)
```
