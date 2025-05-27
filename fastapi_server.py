import os
import time
import traceback
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import requests
from urllib.parse import urlparse

# Import Warden modules
from audio_metrics import AudioMetricsCalculator
from database import init_db
from fastapi.middleware.cors import CORSMiddleware
from web_app import app as web_app


class AudioFileList(BaseModel):
    """Model for list of audio files to analyze in batch mode"""

    file_paths: List[str]  # Can be local file paths or URLs


class MetricsResponse(BaseModel):
    """Model for the metrics response"""

    file_path: str  # Keep for backward compatibility
    filename: str  # Use filename instead of file_name
    latency_points: List[Dict[str, Any]]
    average_latency: float
    p50_latency: float
    p90_latency: float
    min_latency: float
    max_latency: float
    ai_interrupting_user: bool
    user_interrupting_ai: bool
    ai_user_overlap_count: int
    user_ai_overlap_count: int  # Separated the overlap counts
    talk_ratio: float
    average_pitch: float
    words_per_minute: float


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
from url_helper import is_url, download_audio_from_url


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Warden API is running",
        "endpoints": {"web_ui": "http://127.0.0.1:5000", "batch": "/batch"},
    }


@app.post("/batch", response_model=List[MetricsResponse])
def analyze_batch(audio_files: AudioFileList):
    """
    Batch analyze multiple audio files

    Takes a list of file paths or URLs to audio files and returns metrics for each file
    """
    results = []

    print(f"Received batch request for {len(audio_files.file_paths)} files")

    for path in audio_files.file_paths:
        # Check if it's a URL or local file
        local_file_path = path
        print(f"Processing: {path}")  # If it's a URL, download the file first
        if is_url(path):
            try:
                local_file_path = download_audio_from_url(path)
                print(f"Downloaded to: {local_file_path}")
            except Exception as e:
                error_msg = f"Failed to download file from URL: {str(e)}"
                print(f"ERROR: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg,
                )
        # Otherwise check if the local file exists
        elif not os.path.exists(
            os.path.join(calculator.input_dir, path)
        ) and not os.path.exists(path):
            error_msg = f"File not found: {path} (neither as absolute path nor in {calculator.input_dir})"
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
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
        try:
            # Process the file
            print(f"Calling process_file with filename: {filename}")
            metrics = calculator.process_file(filename)
            print(f"process_file successful, received metrics")

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
                        )  # Extract the required metrics
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
            print("Creating MetricsResponse object")
            try:
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
            except Exception as e:
                print(f"ERROR creating MetricsResponse: {str(e)}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise

            results.append(result)
        except Exception as e:
            error_msg = f"Error processing file {local_file_path}: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=error_msg,
            )

    print(f"Batch processing complete, returning {len(results)} results")
    return results


def start_web_app(host="127.0.0.1", port=5000, threads=4):
    """Start web app with Waitress WSGI server"""
    from server import run_flask_app

    # Run the web application with Waitress
    run_flask_app(host, port, threads)


if __name__ == "__main__":
    # For backward compatibility, import and use warden's functionality
    from warden import start_fastapi_server

    start_fastapi_server(host="127.0.0.1", port=8000, start_gui=True)
