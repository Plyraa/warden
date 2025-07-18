"""
Testbed for echo detection.
Processes a directory of audio files, applies echo cancellation,
and reports the segments where echo was detected.
"""
import os
import json
from pathlib import Path
import time
from audio_processor import AudioProcessor
from echo_detection import process_audio_for_echo

def pretty_print_results(filename, results):
    """Prints the echo detection results in a readable format."""
    if not results or not results.get("echo_segments"):
        print(f"\n--- No echo detected for {filename} ---")
        return

    echo_segments = results["echo_segments"]
    total_duration = sum(seg['end'] - seg['start'] for seg in echo_segments)

    print(f"\n--- Echo Detection Report for: {filename} ---")
    print(f"  Total echo duration: {total_duration:.2f} seconds")
    print(f"  Echo cancelled file saved to: {results['echo_cancelled_path']}")
    
    if echo_segments:
        print("  Detected echo segments:")
        for i, seg in enumerate(echo_segments):
            print(f"    {i+1}. Start: {seg['start']:.2f}s, End: {seg['end']:.2f}s, Duration: {seg['end'] - seg['start']:.2f}s")
    print("--------------------------------------------------")


def main():
    # Directories
    input_dir = Path("echoes/")
    temp_dir = Path("temp_downloads/")
    output_dir = Path("echo_cancelled_audio/")

    # Create directories if they don't exist
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Initialize the audio processor
    audio_processor = AudioProcessor(audio_dir=temp_dir)

    # Get list of audio files to process
    audio_files = [f for f in input_dir.iterdir() if f.suffix in (".wav", ".mp3")]

    print(f"Found {len(audio_files)} audio files to process in {input_dir}")

    for audio_path in audio_files:
        try:
            print(f"\n>>> Processing file: {audio_path.name} <<<")
            start_time = time.time()

            # 1. Denoise the audio using the existing AudioProcessor
            # This saves a processed file in `temp_dir`
            print("Step 1: Applying noise reduction...")
            _, _, processed_path = audio_processor.load_and_process_audio(str(audio_path))
            print(f"Noise reduction complete. Processed file at: {processed_path}")

            # 2. Apply echo cancellation and detection on the denoised file
            print("Step 2: Applying echo cancellation and detection...")
            echo_results = process_audio_for_echo(processed_path, output_dir)
            
            # 3. Print the results
            pretty_print_results(audio_path.name, echo_results)

            end_time = time.time()
            print(f"Finished processing {audio_path.name} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"Could not process {audio_path.name}. Error: {e}")

if __name__ == "__main__":
    main()
