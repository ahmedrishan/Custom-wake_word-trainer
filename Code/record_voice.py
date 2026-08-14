"""record_voice.py

Interactive helper to record:
  1. Fixed-length (2.5s) clips of each wake-word phrase, saved flat into
     data/my-voice/ exactly as inject_custom_data.py expects
     (e.g. sherlock_001.wav, hey_sherlock_001.wav, ...)
  2. A longer ambient room-noise recording into data/my-room-noise/
     (room_noise_<timestamp>.wav)

Controls while recording a phrase (shown before every clip):
    [ENTER]  Record this clip (starts immediately, auto-stops after 2.5s)
    g        Continue recording (same as ENTER — explicit "go" command)
    r        Redo — re-record and overwrite the LAST clip you saved
    p        Previous — step back one clip number (to fix/redo an earlier take)
    s        Skip this clip number without recording (leaves a numbering gap)
    f        Finish this phrase now and move to the next one
    q        Quit the whole session (progress already saved is kept)

Usage:
    python record_voice.py
"""

import os
import sys
import time
import wave


def check_dependencies():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print("[!] sounddevice is not installed. Installing sounddevice...")
        os.system(f"{sys.executable} -m pip install sounddevice numpy")
        import sounddevice as sd
        return sd


SAMPLE_RATE = 16000
CLIP_DURATION = 2.5  # seconds, fixed length per requirement #2

VOICE_DIR = os.path.join(".", "data", "my-voice")          # flat — required by inject_custom_data.py
NOISE_DIR = os.path.join(".", "data", "my-room-noise")

PHRASES = ["sherlock", "hey sherlock", "hi sherlock", "hello sherlock"]
CLIPS_PER_PHRASE = 30

CONTROLS_LEGEND = (
    "\n  Controls:  [ENTER]/g = record next   r = redo last   p = previous\n"
    "             s = skip clip   f = finish phrase   q = quit session\n"
)


def record_fixed_clip(output_filepath: str, duration: float = CLIP_DURATION) -> None:
    """Records a fixed-duration clip: manual start via Enter, automatic stop."""
    sd = check_dependencies()
    print(f"\n[+] Recording to {os.path.basename(output_filepath)} for {duration}s...")
    print("    GET READY... 3... 2... 1... SPEAK NOW!")

    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()  # blocks until the fixed duration elapses — automatic stop
    print("    [\u2713] Recording complete!")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with wave.open(output_filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"    Saved: {output_filepath}")


def record_voice_clips() -> None:
    os.makedirs(VOICE_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("        PERSONAL VOICE RECORDING ASSISTANT")
    print("=" * 60)
    print(f"Goal: Record {CLIPS_PER_PHRASE} clips for each phrase, {CLIP_DURATION}s each.")
    print("Tips: Vary your distance (1-5 ft), speed, volume, and emotion.")

    for phrase in PHRASES:
        clean_phrase = phrase.replace(" ", "_")
        print(f"\n---> Target Phrase: '{phrase.upper()}' <---")

        existing = sorted(
            f for f in os.listdir(VOICE_DIR)
            if f.startswith(clean_phrase + "_") and f.endswith(".wav")
        )
        index = len(existing) + 1
        last_saved_path = existing[-1] if existing else None
        if last_saved_path:
            last_saved_path = os.path.join(VOICE_DIR, last_saved_path)
            print(f"  Found {len(existing)} existing clips, continuing from #{index}.")

        while index <= CLIPS_PER_PHRASE:
            print(CONTROLS_LEGEND)
            choice = input(
                f"[{phrase}] Clip {index}/{CLIPS_PER_PHRASE} — press a key: "
            ).strip().lower()

            if choice == "q":
                print("Stopping recording session. Progress saved.")
                return

            if choice == "f":
                print(f"  Finishing '{phrase}' early, moving to next phrase.")
                break

            if choice == "s":
                print(f"  Skipped clip #{index} (no file recorded for this slot).")
                index += 1
                continue

            if choice == "p":
                if index > 1:
                    index -= 1
                    print(f"  Moved back to clip #{index}. Next recording will overwrite it.")
                else:
                    print("  Already at the first clip — can't go further back.")
                continue

            if choice == "r":
                if last_saved_path is None:
                    print("  Nothing recorded yet for this phrase — nothing to redo.")
                    continue
                print(f"  Redoing last clip: {os.path.basename(last_saved_path)}")
                record_fixed_clip(last_saved_path)
                continue

            if choice in ("", "g"):
                filepath = os.path.join(VOICE_DIR, f"{clean_phrase}_{index:03d}.wav")
                record_fixed_clip(filepath)
                last_saved_path = filepath
                index += 1
                continue

            print(f"  Unrecognized key '{choice}'. Use one of the controls shown above.")

    print("\nAll phrases recorded (or skipped).")


def record_room_noise() -> None:
    os.makedirs(NOISE_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("        ROOM BACKGROUND NOISE RECORDING")
    print("=" * 60)
    print("Goal: Record 10-15 minutes of ambient room noise.")
    print("Include typing, laptop fan, room fan, AC, background room sounds.")

    prompt = input("\nHow many minutes of room noise would you like to record? [Default: 10]: ")
    try:
        minutes = float(prompt.strip()) if prompt.strip() else 10.0
    except ValueError:
        minutes = 10.0

    duration_sec = int(minutes * 60)
    filepath = os.path.join(NOISE_DIR, f"room_noise_{int(time.time())}.wav")  # matches system's expected pattern

    print(f"\n[+] Starting continuous recording of {minutes} minutes ({duration_sec} seconds)...")
    record_fixed_clip(filepath, duration=duration_sec)
    print("\n[\u2713] Ambient room noise successfully recorded!")


def main():
    print("=" * 60)
    print("       SHERLOCK WAKE-WORD DATA COLLECTION TOOL")
    print("=" * 60)
    print("1. Record Voice Clips ('sherlock', 'hey sherlock', etc.)")
    print("2. Record Room Ambient Noise")
    print("3. Exit")

    choice = input("\nSelect an option (1-3): ").strip()
    if choice == "1":
        record_voice_clips()
    elif choice == "2":
        record_room_noise()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()