import os
import sys
import time
import wave
import numpy as np

def check_dependencies():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print("[!] sounddevice is not installed. Installing sounddevice...")
        os.system(f"{sys.executable} -m pip install sounddevice numpy")
        import sounddevice as sd
        return sd

def record_audio_clip(output_filepath, duration=2.5, sample_rate=16000):
    sd = check_dependencies()
    print(f"\n[+] Recording to {output_filepath} for {duration} seconds...")
    print("    GET READY... 3... 2... 1... SPEAK NOW!")
    
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("    [✓] Recording complete!")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with wave.open(output_filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit PCM = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    print(f"    Saved: {output_filepath}")

def record_voice_clips():
    phrases = ["sherlock", "hey sherlock", "hi sherlock", "hello sherlock"]
    save_dir = os.path.join(".", "data", "my-voice")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("        PERSONAL VOICE RECORDING ASSISTANT")
    print("="*60)
    print("Goal: Record 20-30 clips for each phrase.")
    print("Tips: Vary your distance (1-5 ft), speed, volume, and emotion.")
    
    for phrase in phrases:
        clean_phrase = phrase.replace(" ", "_")
        print(f"\n---> Target Phrase: '{phrase.upper()}' <---")
        
        # Check existing clips count
        existing = [f for f in os.listdir(save_dir) if f.startswith(clean_phrase) and f.endswith(".wav")]
        count = len(existing) + 1
        
        while True:
            prompt = input(f"\nPress [ENTER] to record clip #{count} for '{phrase}' (or type 'next' to move to next phrase, 'q' to quit): ")
            if prompt.strip().lower() == 'q':
                return
            if prompt.strip().lower() == 'next':
                break
                
            filepath = os.path.join(save_dir, f"{clean_phrase}_{count:03d}.wav")
            record_audio_clip(filepath, duration=2.5)
            count += 1

def record_room_noise():
    save_dir = os.path.join(".", "data", "my-room-noise")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("        ROOM BACKGROUND NOISE RECORDING")
    print("="*60)
    print("Goal: Record 10-15 minutes of ambient room noise.")
    print("Include typing, laptop fan, room fan, AC, background room sounds.")
    
    prompt = input("\nHow many minutes of room noise would you like to record? [Default: 10]: ")
    try:
        minutes = float(prompt.strip()) if prompt.strip() else 10.0
    except ValueError:
        minutes = 10.0
        
    duration_sec = int(minutes * 60)
    filepath = os.path.join(save_dir, f"room_noise_{int(time.time())}.wav")
    
    print(f"\n[+] Starting continuous recording of {minutes} minutes ({duration_sec} seconds)...")
    record_audio_clip(filepath, duration=duration_sec)
    print("\n[✓] Ambient room noise successfully recorded!")

def main():
    print("="*60)
    print("       SHERLOCK WAKE-WORD DATA COLLECTION TOOL")
    print("="*60)
    print("1. Record Voice Clips ('sherlock', 'hey sherlock', etc.)")
    print("2. Record Room Ambient Noise")
    print("3. Exit")
    
    choice = input("\nSelect an option (1-3): ").strip()
    if choice == '1':
        record_voice_clips()
    elif choice == '2':
        record_room_noise()
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
