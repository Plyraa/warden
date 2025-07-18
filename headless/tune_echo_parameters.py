"""
Testbed for tuning echo detection parameters on a specific file.
"""
import os
import json
from pathlib import Path
import time
from audio_processor import AudioProcessor
from echo_detection import process_audio_for_echo

def pretty_print_results(filename, results, params):
    """Prints the echo detection results in a readable format."""
    if not results or not results.get("echo_segments"):
        print(f"\n--- No echo detected for {filename} with params: {params} ---")
        return

    echo_segments = results["echo_segments"]
    total_duration = sum(seg['end'] - seg['start'] for seg in echo_segments)

    print(f"\n--- Echo Detection Report for: {filename} ---")
    print(f"  Parameters: {params}")
    print(f"  Total echo duration: {total_duration:.2f} seconds")
    
    if echo_segments:
        print("  Detected echo segments:")
        for i, seg in enumerate(echo_segments):
            print(f"    {i+1}. Start: {seg['start']:.2f}s, End: {seg['end']:.2f}s, Duration: {seg['end'] - seg['start']:.2f}s")
    print("--------------------------------------------------")


def main():
    # File to test
    test_file = Path("echoes/echo019731edffc67ed980ff0a4eeea4fe796a6e.mp3")

    # Directories
    temp_dir = Path("temp_downloads/")
    output_dir = Path("echo_cancelled_audio/")

    # Create directories if they don't exist
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Initialize the audio processor
    audio_processor = AudioProcessor(audio_dir=temp_dir)

    # --- Parameter Sets to Test ---
    # You can add more dictionaries to this list to test different combinations.
    parameter_sets = [
        {"reduction_threshold": 0.5, "correlation_threshold": 0.02},
        {"reduction_threshold": 0.5, "correlation_threshold": 0.04},
        {"reduction_threshold": 0.5, "correlation_threshold": 0.05},
        {"reduction_threshold": 0.66, "correlation_threshold": 0.02},
        {"reduction_threshold": 0.66, "correlation_threshold": 0.04},
        {"reduction_threshold": 0.66, "correlation_threshold": 0.05},
        {"reduction_threshold": 0.83, "correlation_threshold": 0.05},
        {"reduction_threshold": 1.0, "correlation_threshold": 0.05},
        {"reduction_threshold": 1.0, "correlation_threshold": 0.02},
        {"reduction_threshold": 1.0, "correlation_threshold": 0.04},
        #{"reduction_threshold": 1.0, "correlation_threshold": 0.10},
        {"reduction_threshold": 1.5, "correlation_threshold": 0.05},
        #{"reduction_threshold": 1.5, "correlation_threshold": 0.10},
        {"reduction_threshold": 2.0, "correlation_threshold": 0.05},
        #{"reduction_threshold": 2.0, "correlation_threshold": 0.10},
        #{"reduction_threshold": 2.0, "correlation_threshold": 0.15},
        #{"reduction_threshold": 2.0, "correlation_threshold": 0.20},
        #{"reduction_threshold": 2.0, "correlation_threshold": 0.25},
        #{"reduction_threshold": 2.0, "correlation_threshold": 0.30},
        #{"reduction_threshold": 3.0, "correlation_threshold": 0.10},
        #{"reduction_threshold": 3.0, "correlation_threshold": 0.20},
        #{"reduction_threshold": 3.0, "correlation_threshold": 0.30},
        #{"reduction_threshold": 3.0, "correlation_threshold": 0.40},
        #{"reduction_threshold": 3.0, "correlation_threshold": 0.50},
        #{"reduction_threshold": 3.5, "correlation_threshold": 0.35},
        #{"reduction_threshold": 4.0, "correlation_threshold": 0.40},
    ]

    6-8
    19-21
    23-24
    27-30
    32-34
    39-42
    48-50
    53-55
    58-60
    65-67
    69-71
    75-77
    79-81

    print(f"--- Starting Parameter Tuning for: {test_file.name} ---")

    try:
        # 1. Denoise the audio once
        print("Step 1: Applying noise reduction (once)...")
        _, _, processed_path = audio_processor.load_and_process_audio(str(test_file))
        print(f"Noise reduction complete. Processed file at: {processed_path}")

        # 2. Iterate through parameter sets and test each one
        for params in parameter_sets:
            print(f"\n>>> Testing with parameters: {params} <<<")
            start_time = time.time()

            # Apply echo cancellation and detection with the current set of parameters
            echo_results = process_audio_for_echo(processed_path, output_dir, params=params)
            
            # Print the results for this set of parameters
            pretty_print_results(test_file.name, echo_results, params)

            end_time = time.time()
            print(f"Finished test run in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Could not process {test_file.name}. Error: {e}")

if __name__ == "__main__":
    main()
