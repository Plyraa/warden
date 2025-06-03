import argparse
import os
import multiprocessing
from waitress import serve
import uvicorn

from audio_metrics import AudioMetricsCalculator
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


def run_flask_app(host="127.0.0.1", port=5000, threads=4):
    """
    This function serves the Flask application using Waitress

    Args:
        host: Host address to bind to
        port: Port to listen on
        threads: Number of worker threads
    """

    from web_app import app

    # This allows Waitress to import the application object
    flask_app = app
    
    # Use Waitress to serve the Flask app
    serve(flask_app, host=host, port=port, threads=threads)


def run_fastapi_app(host="127.0.0.1", port=8000):
    """
    Run the FastAPI application using Uvicorn

    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    # Import here to avoid circular imports
    from fastapi_server import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port)


def run_combined(host="127.0.0.1", api_port=8000, web_port=5000, threads=4):
    """
    Start both FastAPI and Flask web UI servers using multiprocessing

    Args:
        host: Host address for both servers
        api_port: Port for FastAPI server
        web_port: Port for Flask web UI
        threads: Number of threads for Waitress
    """
    # Start web UI in a separate process
    web_process = multiprocessing.Process(
        target=run_flask_app, args=(host, web_port, threads)
    )
    web_process.daemon = True
    web_process.start()

    print(f"Started Web UI at http://{host}:{web_port}")
    print(f"Starting FastAPI server at http://{host}:{api_port}")

    # Run FastAPI in the main process
    run_fastapi_app(host, api_port)


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
        "--process",
        action="store_true",
        help="Process audio files instead of starting servers",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Start only the Flask web UI",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for servers")
    parser.add_argument(
        "--api-port", type=int, default=8000, help="Port for FastAPI server"
    )
    parser.add_argument(
        "--web-port", type=int, default=5000, help="Port for Flask web UI"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of Waitress worker threads"
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
    elif args.web_only:
        # Start only the Flask web UI
        print(f"Starting Web UI only at http://{args.host}:{args.web_port}")
        run_flask_app(args.host, args.web_port, args.threads)
    elif args.api_only:
        # Start only the FastAPI server
        print(f"Starting FastAPI server only at http://{args.host}:{args.api_port}")
        run_fastapi_app(args.host, args.api_port)
    else:
        # Start both by default (with multiprocessing)
        print("Starting both FastAPI server and Web UI")
        print(f"- FastAPI: http://{args.host}:{args.api_port}")
        print(f"- Web UI:  http://{args.host}:{args.web_port}")
        run_combined(args.host, args.api_port, args.web_port, args.threads)


if __name__ == "__main__":
    init_db()  # Initialize the database and create tables if they don't exist
    main()
