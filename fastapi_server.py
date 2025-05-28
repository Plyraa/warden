import os
import traceback
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from url_helper import is_url, download_audio_from_url

# Import Warden modules
from audio_metrics import AudioMetricsCalculator
from database import init_db, SessionLocal
from fastapi.middleware.cors import CORSMiddleware


class AudioFileList(BaseModel):
    """Model for list of audio files to analyze in batch mode"""

    file_paths: List[str]  # Can be local file paths or URLs


class MetricsResponse(BaseModel):
    """Model for the metrics response"""

    file_path: str  # Keep for backward compatibility
    filename: str  # Use filename instead of file_name
    status: str  # "success" or "error"
    error_message: str = None  # Only present if status is "error"
    latency_points: List[Dict[str, Any]] = []
    average_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    ai_interrupting_user: bool = False
    user_interrupting_ai: bool = False
    ai_user_overlap_count: int = 0
    user_ai_overlap_count: int = 0  # Separated the overlap counts
    talk_ratio: float = 0.0
    average_pitch: float = 0.0
    words_per_minute: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database when app starts
    init_db()

    # Startup: nothing else to do
    yield

    # Shutdown: nothing to clean up


app = FastAPI(
    lifespan=lifespan,
    title="Warden API",
    description="API for Warden audio analysis tool",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create calculator instance
calculator = AudioMetricsCalculator()


# Import the enhanced URL helpers


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Warden API is running",
        "endpoints": {"web_ui": "http://127.0.0.1:5000", "batch": "/batch"},
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Warden API",
        "version": "1.0.0",
    }


@app.post("/batch", response_model=List[MetricsResponse])
def analyze_batch(audio_files: AudioFileList):
    """
    Batch analyze multiple audio files

    Takes a list of file paths or URLs to audio files and returns metrics for each file.
    Returns partial results even if some files fail - each file gets a status indicator.
    """
    results = []

    print(f"Received batch request for {len(audio_files.file_paths)} files")

    for path in audio_files.file_paths:
        print(f"Processing: {path}")

        try:
            # Check if it's a URL or local file
            local_file_path = path
            source_url = None

            if is_url(path):
                source_url = path
                # First check if we already have analysis for this URL
                db_session = SessionLocal()
                try:
                    from database import get_analysis_by_url, recreate_metrics_from_db

                    existing_analysis = get_analysis_by_url(db_session, source_url)
                    if existing_analysis:
                        # We already have this URL analyzed
                        print(f"Found existing analysis for URL: {source_url}")
                        existing_metrics = recreate_metrics_from_db(existing_analysis)
                        if existing_metrics:
                            results.append(
                                MetricsResponse(
                                    file_path=path,
                                    filename=existing_metrics["filename"],
                                    status="success",
                                    **{
                                        k: v
                                        for k, v in existing_metrics[
                                            "latency_metrics"
                                        ].items()
                                        if v is not None
                                    },
                                    **{
                                        f"vad_{k}": v
                                        for k, v in existing_metrics[
                                            "vad_latency_metrics"
                                        ].items()
                                        if v is not None
                                    },
                                    ai_interrupting_user=existing_metrics.get(
                                        "ai_interrupting_user"
                                    ),
                                    user_interrupting_ai=existing_metrics.get(
                                        "user_interrupting_ai"
                                    ),
                                    talk_ratio=existing_metrics.get("talk_ratio"),
                                    average_pitch_hz=existing_metrics.get(
                                        "average_pitch_hz"
                                    ),
                                    words_per_minute=existing_metrics.get(
                                        "words_per_minute"
                                    ),
                                )
                            )
                            continue
                finally:
                    db_session.close()

                # If no existing analysis, download the file
                try:
                    local_file_path = download_audio_from_url(path)
                    print(f"Downloaded to: {local_file_path}")
                except Exception as e:
                    error_msg = f"Failed to download file from URL: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    # Create error response for this file and continue
                    results.append(
                        MetricsResponse(
                            file_path=path,
                            filename=os.path.basename(path),
                            status="error",
                            error_message=error_msg,
                        )
                    )
                    continue
            # Otherwise check if the local file exists
            elif not os.path.exists(
                os.path.join(calculator.input_dir, path)
            ) and not os.path.exists(path):
                error_msg = f"File not found: {path} (neither as absolute path nor in {calculator.input_dir})"
                print(f"ERROR: {error_msg}")
                # Create error response for this file and continue
                results.append(
                    MetricsResponse(
                        file_path=path,
                        filename=os.path.basename(path),
                        status="error",
                        error_message=error_msg,
                    )
                )
                continue
            else:
                if os.path.exists(path):
                    print(f"Found file as absolute path: {path}")
                else:
                    print(
                        f"Found file in input directory: {os.path.join(calculator.input_dir, path)}"
                    )

            # Get filename from path
            filename = os.path.basename(local_file_path)
            print(f"Using filename: {filename}")
            # Process the file
            print(f"Calling process_file with filename: {filename}")
            metrics = calculator.process_file(filename, source_url=source_url)
            print("process_file successful, received metrics")

            # Extract the latency points
            latency_points = []
            if metrics.get("vad_latency_details"):
                print(f"Found {len(metrics['vad_latency_details'])} latency details")
                for point in metrics["vad_latency_details"]:
                    if point["interaction_type"] == "user_to_agent":
                        latency_ms = point.get("latency_ms", 0)
                        # If latency_ms isn't available, try to get latency_seconds and convert
                        if latency_ms == 0 and "latency_seconds" in point:
                            latency_ms = point["latency_seconds"] * 1000

                        latency_points.append(
                            {
                                "latency_ms": latency_ms,
                                "moment": point["to_turn_start"],
                            }
                        )

            # Extract the required metrics
            print("Extracting metrics for response")
            latency_metrics = metrics.get("vad_latency_metrics", {})
            print(
                f"VAD latency metrics: {list(latency_metrics.keys()) if latency_metrics else 'None'}"
            )

            overlap_data = metrics.get("overlap_data", {})
            print(
                f"Overlap data keys: {list(overlap_data.keys()) if overlap_data else 'None'}"
            )

            # Count the overlaps by type
            ai_user_overlap_count = 0
            user_ai_overlap_count = 0

            overlaps = overlap_data.get("overlaps", [])
            print(f"Found {len(overlaps)} overlaps")

            for overlap in overlaps:
                if overlap.get("interrupter") == "ai_agent":
                    ai_user_overlap_count += 1
                elif overlap.get("interrupter") == "user":
                    user_ai_overlap_count += 1

            print(
                f"Counted {ai_user_overlap_count} AI interruptions and {user_ai_overlap_count} user interruptions"
            )
            print("Creating successful MetricsResponse object")

            # Prepare values with appropriate error handling
            avg_latency = latency_metrics.get("avg_latency", 0)
            p50_latency = latency_metrics.get("p50_latency", 0)
            p90_latency = latency_metrics.get("p90_latency", 0)
            min_latency = latency_metrics.get("min_latency", 0)
            max_latency = latency_metrics.get("max_latency", 0)

            print(
                f"Latency values (seconds): avg={avg_latency}, p50={p50_latency}, p90={p90_latency}"
            )

            result = MetricsResponse(
                file_path=path,  # Use the original path/URL that was passed in
                filename=filename,  # Include the filename for consistency
                status="success",
                latency_points=latency_points,
                average_latency=avg_latency * 1000,  # Convert to ms
                p50_latency=p50_latency * 1000,  # Convert to ms
                p90_latency=p90_latency * 1000,  # Convert to ms
                min_latency=min_latency * 1000,  # Convert to ms
                max_latency=max_latency * 1000,  # Convert to ms
                ai_interrupting_user=metrics.get("ai_interrupting_user", False),
                user_interrupting_ai=metrics.get("user_interrupting_ai", False),
                ai_user_overlap_count=ai_user_overlap_count,  # AI interrupting user count
                user_ai_overlap_count=user_ai_overlap_count,  # User interrupting AI count
                talk_ratio=metrics.get("talk_ratio", 0),
                average_pitch=metrics.get("average_pitch", 0),
                words_per_minute=metrics.get("words_per_minute", 0),
            )
            print("MetricsResponse object created successfully")
            results.append(result)

        except Exception as e:
            error_msg = f"Error processing file {path}: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")

            # Create error response for this file and continue processing other files
            results.append(
                MetricsResponse(
                    file_path=path,
                    filename=os.path.basename(path),
                    status="error",
                    error_message=error_msg,
                )
            )

    print(f"Batch processing complete, returning {len(results)} results")
    return results


def start_web_app(host="127.0.0.1", port=5000, threads=4):
    """Start web app with Waitress WSGI server"""
    from warden import run_flask_app

    # Run the web application with Waitress
    run_flask_app(host, port, threads)


if __name__ == "__main__":
    # For backward compatibility, import and use warden's functionality
    from warden import run_combined

    # Start both FastAPI and Flask web UI
    run_combined(host="127.0.0.1", api_port=8000, web_port=5000, threads=4)
