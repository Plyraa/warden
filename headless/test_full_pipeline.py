"""
Testbed for running the full audio processing pipeline and logging results.

This script processes a directory of audio files using the main AudioProcessor,
prints key findings for each file, and saves the complete metrics to a CSV file.
"""
import os
import csv
from pathlib import Path
import time
import json
from audio_processor import AudioProcessor

def main():
    # Directories
    input_dir = Path("echoes/")
    temp_dir = Path("temp_downloads/")
    csv_dir = Path("csv_outputs/")

    # Create directories if they don't exist
    temp_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    # Initialize the audio processor
    audio_processor = AudioProcessor(audio_dir=temp_dir)

    # Get list of audio files to process
    audio_files = [f for f in input_dir.iterdir() if f.suffix in (".wav", ".mp3")]
    print(f"Found {len(audio_files)} audio files to process in {input_dir}")

    all_results = []

    for audio_path in audio_files:
        try:
            print(f"\n{'='*20} Processing: {audio_path.name} {'='*20}")
            start_time = time.time()

            # Run the full pipeline
            metrics = audio_processor.process_file(str(audio_path))
            
            if metrics:
                # Print key results to the console for immediate feedback
                print(f"\n--- Analysis Summary for {audio_path.name} ---")
                print(f"  Heavy Echo Detected: {metrics.get('hasHeavyEcho')}")
                print(f"  Light Echo Detected: {metrics.get('hasLightEcho')}")
                print(f"  Total Echo Duration: {metrics.get('total_echo_duration', 0):.2f}s")
                
                if metrics.get('echo_segments'):
                    print("  Detected Echo Segments:")
                    for i, seg in enumerate(metrics['echo_segments']):
                        print(f"    {i+1}. Start: {seg['start']:.2f}s, End: {seg['end']:.2f}s")
                
                if metrics.get('heavy_echo_segments'):
                    print("  Heavy Echo Triggering Events:")
                    for i, event in enumerate(metrics['heavy_echo_segments']):
                        agent_end = event.get('agent_turn_end_time', 0)
                        false_user_seg = event.get('triggering_false_user_segment', {})
                        start = false_user_seg.get('start', 0)
                        end = false_user_seg.get('end', 0)
                        print(f"    {i+1}. Agent stopped at {agent_end:.2f}s during false user speech from {start:.2f}s to {end:.2f}s")

                all_results.append(metrics)

            end_time = time.time()
            print(f"\nFinished processing {audio_path.name} in {end_time - start_time:.2f} seconds.")
            print(f"{'='* (42 + len(audio_path.name))}")


        except Exception as e:
            print(f"Could not process {audio_path.name}. Error: {e}")

    # Save all results to a single CSV file
    if all_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_csv_path = csv_dir / f"full_pipeline_analysis_{timestamp}.csv"
        
        # Flatten the nested dictionaries for CSV writing
        flat_results = []
        for res in all_results:
            flat_res = {
                "filename": res.get("filename"),
                "hasHeavyEcho": res.get("hasHeavyEcho"),
                "hasLightEcho": res.get("hasLightEcho"),
                "total_echo_duration": res.get("total_echo_duration"),
                "heavy_echo_segments": json.dumps(res.get("heavy_echo_segments")),
                "ai_interrupting_user": res.get("ai_interrupting_user"),
                "user_interrupting_ai": res.get("user_interrupting_ai"),
                "talk_ratio": res.get("talk_ratio"),
                "average_pitch": res.get("average_pitch"),
                "words_per_minute": res.get("words_per_minute"),
                "avg_latency": res.get("vad_latency_metrics", {}).get("avg_latency"),
                "p50_latency": res.get("vad_latency_metrics", {}).get("p50_latency"),
                "p90_latency": res.get("vad_latency_metrics", {}).get("p90_latency"),
                "total_overlap_count": res.get("overlap_data", {}).get("total_overlap_count"),
                "echo_segments": json.dumps(res.get("echo_segments")),
                "vad_latency_details": json.dumps(res.get("vad_latency_details")),
                "original_path": res.get("original_path"),
                "processed_path": res.get("processed_path"),
            }
            flat_results.append(flat_res)

        headers = flat_results[0].keys()
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flat_results)
        
        print(f"\nSuccessfully processed {len(all_results)} files.")
        print(f"Full analysis saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
