import argparse
import os
import multiprocessing

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


def run_flask_app(host, port):
    """
    Function to run the Flask app in a multiprocessing context
    
    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    flask_app.run(host=host, port=port, debug=False)

def start_fastapi_server(host="127.0.0.1", port=8000, start_gui=False):
    """
    Start the FastAPI server with optional web UI
    
    Args:
        host: Host address for FastAPI server
        port: Port for FastAPI server
        start_gui: Whether to also start the Flask web UI
    """
    # Import here to avoid circular imports
    from fastapi_server import app
    import uvicorn

    # Start Flask web UI in a separate process if requested
    if start_gui:
        print(f"Starting Web UI at http://{host}:5000")
        flask_process = multiprocessing.Process(
            target=run_flask_app,
            args=(host, 5000)
        )
        flask_process.daemon = True
        flask_process.start()

    # Start FastAPI server
    print(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


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
        "--gui",
        action="store_true",
        help="Start web interface for visualization alongside FastAPI",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process audio files instead of starting servers",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Legacy option: Start only web interface without FastAPI",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for servers")
    parser.add_argument(
        "--api-port", type=int, default=8000, help="Port for FastAPI server"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for web interface (legacy)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {os.path.abspath(args.output_dir)}")

    # Choose operation mode
    if args.process:
        # Process audio files mode
        print("Processing audio files")
        process_audio_files(args.input_dir, args.output_dir)
    elif args.web:
        # Legacy mode: only web UI
        print("Starting web interface only (legacy mode)")
        start_web_interface(args.host, args.port)
    else:
        # Default mode: start FastAPI server with optional GUI
        start_fastapi_server(args.host, args.api_port, args.gui)


if __name__ == "__main__":
    init_db()  # Initialize the database and create tables if they don't exist
    main()
