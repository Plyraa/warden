import argparse
import os

from audio_metrics import AudioMetricsCalculator
from web_ui import app as flask_app
from database import init_db  # Added to initialize DB on startup


def process_audio_files(input_dir="stereo_test_calls", output_dir="sampled_test_calls"):
    """
    Process all audio files in the input directory

    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed audio files
    """
    # Create calculator instance
    calculator = AudioMetricsCalculator(input_dir, output_dir)

    # Get list of audio files
    audio_files = [
        f for f in os.listdir(input_dir) if f.endswith(".mp3") or f.endswith(".wav")
    ]

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) to process")

    # Process each file
    for i, filename in enumerate(audio_files):
        print(f"Processing file {i + 1}/{len(audio_files)}: {filename}")

        # Calculate metrics
        metrics = calculator.process_file(filename)

        # Print summary of metrics
        print_metrics_summary(metrics)

    print(f"All files processed. Downsampled files saved to {output_dir}")


def print_metrics_summary(metrics):
    """Print a summary of the calculated metrics"""
    print("\nMetrics Summary:")
    print(f"File: {metrics['filename']}")
    print(f"Downsampled to: {metrics['downsampled_path']}")

    # Latency metrics
    latency = metrics["latency_metrics"]
    print(f"Average Latency: {latency['avg_latency']:.2f} ms")
    print(
        f"P10/P50/P90 Latency: {latency['p10_latency']:.2f}/{latency['p50_latency']:.2f}/{latency['p90_latency']:.2f} ms"
    )

    # Other metrics
    print(f"AI Interrupting User: {'Yes' if metrics['ai_interrupting_user'] else 'No'}")
    print(f"User Interrupting AI: {'Yes' if metrics['user_interrupting_ai'] else 'No'}")
    print(f"Talk Ratio (Agent/User): {metrics['talk_ratio']:.2f}")
    print(f"Average Pitch: {metrics['average_pitch']:.2f} Hz")
    print(f"Words Per Minute: {metrics['words_per_minute']:.2f} WPM")

    # Count of speech segments
    print(f"User Speech Segments: {len(metrics['user_windows'])}")
    print(f"Agent Speech Segments: {len(metrics['agent_windows'])}")

    # Transcript
    if metrics.get("transcript_data") and metrics["transcript_data"].get("dialog"):
        print("\nTranscript:")
        print(metrics["transcript_data"]["dialog"])
    else:
        print("\nTranscript: Not available.")
    print("-" * 50)


def start_web_interface(host="127.0.0.1", port=5000):
    """Start the web interface"""
    print(f"Starting web interface at http://{host}:{port}")
    flask_app.run(host=host, port=port, debug=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Audio Analysis for Jotform AI Agent Calls"
    )
    parser.add_argument(
        "--input-dir",
        default="stereo_test_calls",
        help="Directory containing input audio files",
    )
    parser.add_argument(
        "--output-dir",
        default="sampled_test_calls",
        help="Directory to save processed audio files",
    )
    parser.add_argument(
        "--web", action="store_true", help="Start web interface for visualization"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address for web interface"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for web interface")

    args = parser.parse_args()

    # Process audio files only if --web is not specified
    if not args.web:
        process_audio_files(args.input_dir, args.output_dir)
    else:
        # Ensure output directory exists for web UI even if not pre-processing
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        print(
            f"Output directory for web processing: {os.path.abspath(args.output_dir)}"
        )
        # Pass input and output dirs to the web app context if needed,
        # or ensure the web_ui.py uses these defaults or configured paths.
        # For now, web_ui.py uses its own defaults which match.
        start_web_interface(args.host, args.port)


if __name__ == "__main__":
    init_db()  # Initialize the database and create tables if they don't exist
    main()
