import os
import time
import sounddevice as sd
import soundfile as sf

OUTPUT_FOLDER = "no_cry"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 7
CLIP_COUNT = 300

def record_no_cry_clips():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Find the highest existing clip number
    existing_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("no_cry_") and f.endswith(".wav")]
    existing_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files if f.split("_")[2].split(".")[0].isdigit()]
    start_number = max(existing_numbers, default=0) + 1

    print(f"Recording {CLIP_COUNT} non-cry (background) audio clips.")
    print(f"Destination folder: {OUTPUT_FOLDER}")
    print(f"Starting from clip #{start_number}.")

    for i in range(CLIP_COUNT):
        clip_number = start_number + i
        print(f"\nRecording clip #{clip_number} for {DURATION} seconds...")
        
        # Start recording
        audio_data = sd.rec(
            int(DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype='int16'
        )
        sd.wait()  # Wait until recording is finished

        filename = os.path.join(OUTPUT_FOLDER, f"no_cry_{clip_number}.wav")

        sf.write(filename, audio_data, SAMPLE_RATE)
        print(f"Saved clip #{clip_number} to {filename}")

        time.sleep(2)

    print("\nFinished recording no_cry clips!")

if __name__ == "__main__":
    try:
        record_no_cry_clips()
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
