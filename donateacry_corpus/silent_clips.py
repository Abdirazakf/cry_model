import os
import time
import sounddevice as sd
import soundfile as sf

OUTPUT_FOLDER = "no_cry"
SAMPLE_RATE = 16000       # Sampling rate in Hz
CHANNELS = 1
DURATION = 7              # Record duration for each clip in seconds
CLIP_COUNT = 300           # How many clips to record

def record_no_cry_clips():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Recording {CLIP_COUNT} non-cry (background) audio clips.")
    print(f"Destination folder: {OUTPUT_FOLDER}")

    for i in range(CLIP_COUNT):
        clip_number = i + 1
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
