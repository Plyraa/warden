import os
import time
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
from web_ui import app as flask_app


class AudioFileList(BaseModel):
    """Model for list of audio files to analyze in batch mode"""

    file_paths: List[str]  # Can be local file paths or URLs


class MetricsResponse(BaseModel):
    """Model for the metrics response"""

    file_path: str  # Keep for backward compatibility
    filename: str   # Use filename instead of file_name
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


# Add a URL download function
def is_url(path):
    """Check if a string is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_mp3_from_url(url):
    """
    Download an MP3 file from a URL and save it to a temporary directory

    Args:
        url: URL of the MP3 file

    Returns:
        Path to the downloaded file or None if download failed
    """
    try:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If filename is empty or doesn't end with .mp3, create a unique name
        if not filename or not filename.lower().endswith(".mp3"):
            filename = f"downloaded_{int(time.time())}.mp3"

        # Create downloads directory if it doesn't exist
        download_dir = os.path.join(os.getcwd(), "stereo_test_calls")
        os.makedirs(download_dir, exist_ok=True)

        # Create the save path
        save_path = os.path.join(download_dir, filename)

        # Download the file
        print(f"Downloading {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded file to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error downloading file from {url}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error downloading file from {url}: {str(e)}"
        )


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

    for path in audio_files.file_paths:
        # Check if it's a URL or local file
        local_file_path = path

        # If it's a URL, download the file first
        if is_url(path):
            try:
                local_file_path = download_mp3_from_url(path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download file from URL: {str(e)}",
                )
        # Otherwise check if the local file exists
        elif not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        # Get filename from path
        filename = os.path.basename(local_file_path)

        try:
            # Process the file
            metrics = calculator.process_file(filename)

            # Extract the latency points
            latency_points = []
            if metrics.get("vad_latency_details"):
                for point in metrics["vad_latency_details"]:
                    if point["interaction_type"] == "user_to_agent":
                        latency_points.append(
                            {
                                "latency_ms": point["latency_ms"],
                                "moment": point["to_turn_start"],
                            }
                        )

            # Extract the required metrics
            latency_metrics = metrics.get("vad_latency_metrics", {})
            overlap_data = metrics.get("overlap_data", {})            # Count the overlaps by type
            ai_user_overlap_count = 0
            user_ai_overlap_count = 0
            
            overlaps = overlap_data.get("overlaps", [])
            for overlap in overlaps:
                if overlap.get("interrupter") == "ai_agent":
                    ai_user_overlap_count += 1
                elif overlap.get("interrupter") == "user":
                    user_ai_overlap_count += 1
            
            result = MetricsResponse(
                file_path=path,  # Use the original path/URL that was passed in
                filename=filename,  # Include the filename for consistency
                latency_points=latency_points,
                average_latency=latency_metrics.get("avg_latency", 0)
                * 1000,  # Convert to ms
                p50_latency=latency_metrics.get("p50_latency", 0)
                * 1000,  # Convert to ms
                p90_latency=latency_metrics.get("p90_latency", 0)
                * 1000,  # Convert to ms
                min_latency=latency_metrics.get("min_latency", 0)
                * 1000,  # Convert to ms
                max_latency=latency_metrics.get("max_latency", 0)
                * 1000,  # Convert to ms
                ai_interrupting_user=metrics.get("ai_interrupting_user", False),
                user_interrupting_ai=metrics.get("user_interrupting_ai", False),
                ai_user_overlap_count=ai_user_overlap_count,  # AI interrupting user count
                user_ai_overlap_count=user_ai_overlap_count,  # User interrupting AI count
                talk_ratio=metrics.get("talk_ratio", 0),
                average_pitch=metrics.get("average_pitch", 0),
                words_per_minute=metrics.get("words_per_minute", 0),
            )

            results.append(result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {local_file_path}: {str(e)}",
            )

    return results


def start_flask_app(host="127.0.0.1", port=5000):
    """Start Flask app with specified host and port"""
    flask_app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    # For backward compatibility, import and use warden's functionality
    from warden import start_fastapi_server

    start_fastapi_server(host="127.0.0.1", port=8000, start_gui=True)
