"""
Headless Warden - Pure FastAPI Audio Processing Service
No database, no ElevenLabs, no web UI - just streaming audio analysis
"""
import os
import asyncio
import logging
import traceback
from typing import List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models import AudioFileList, MetricsResponse, HealthResponse
from audio_processor import AudioProcessor
from url_downloader import URLDownloader
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    Config.ensure_temp_dir()
    logger.info("Warden Headless API starting up...")
    
    yield
    
    # Shutdown
    logger.info("Warden Headless API shutting down...")

app = FastAPI(
    title="Warden Headless API",
    description="Headless audio analysis service with streaming batch processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor and downloader
processor = AudioProcessor()
downloader = URLDownloader(
    download_dir=str(Config.TEMP_DIR),
    max_file_size=Config.get_max_file_size_bytes()
)


@app.get("/", response_model=dict)
def read_root():
    """Root endpoint"""
    return {
        "message": "Warden Headless API is running",
        "service": "Warden Headless",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "batch": "/batch",
            "batch_stream": "/batch-stream"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Warden Headless",
        version="1.0.0"
    )


@app.post("/batch", response_model=List[MetricsResponse])
def analyze_batch(audio_files: AudioFileList):
    """
    Batch analyze multiple audio files
    
    Takes a list of file paths or URLs and returns metrics for each file.
    Returns partial results even if some files fail.
    """
    results = []
    downloaded_files = []  # Track downloaded files for cleanup
    print (downloaded_files)
    logger.info(f"Received batch request for {len(audio_files.file_paths)} files")

    for path in audio_files.file_paths:
        logger.info(f"Processing: {path}")
        
        try:
            local_file_path = path
            
            # Handle URL downloads
            if downloader.is_url(path):
                try:
                    local_file_path = downloader.download(path)
                    downloaded_files.append(local_file_path)
                    logger.info(f"Downloaded to: {local_file_path}")
                except Exception as e:
                    error_msg = f"Failed to download file from URL: {str(e)}"
                    logger.error(error_msg)
                    results.append(MetricsResponse(
                        file_path=path,
                        filename=os.path.basename(path),
                        status="error",
                        error_message=error_msg,
                    ))
                    continue
            
            # Check if local file exists
            elif not os.path.exists(local_file_path):
                error_msg = f"File not found: {path}"
                logger.error(error_msg)
                results.append(MetricsResponse(
                    file_path=path,
                    filename=os.path.basename(path),
                    status="error",
                    error_message=error_msg,
                ))
                continue

            # Process the file
            filename = os.path.basename(local_file_path)
            logger.info(f"Processing file: {filename}")
            
            metrics = processor.process_file(local_file_path)
            
            # Extract latency points
            latency_points = []
            if metrics.get("vad_latency_details"):
                for point in metrics["vad_latency_details"]:
                    if point["interaction_type"] == "user_to_agent":
                        latency_ms = point.get("latency_ms", 0)
                        if latency_ms == 0 and "latency_seconds" in point:
                            latency_ms = point["latency_seconds"] * 1000

                        latency_points.append({
                            "latency_ms": latency_ms,
                            "moment": point["to_turn_start"],
                        })

            # Extract metrics
            latency_metrics = metrics.get("vad_latency_metrics", {})
            overlap_data = metrics.get("overlap_data", {})

            # Count overlaps
            ai_user_overlap_count = 0
            user_ai_overlap_count = 0
            overlaps = overlap_data.get("overlaps", [])
            
            for overlap in overlaps:
                if overlap.get("interrupter") == "ai_agent":
                    ai_user_overlap_count += 1
                elif overlap.get("interrupter") == "user":
                    user_ai_overlap_count += 1

            # Create successful response
            result = MetricsResponse(
                file_path=path,
                filename=filename,
                status="success",
                latency_points=latency_points,
                average_latency=latency_metrics.get("avg_latency", 0) * 1000,
                p50_latency=latency_metrics.get("p50_latency", 0) * 1000,
                p90_latency=latency_metrics.get("p90_latency", 0) * 1000,
                min_latency=latency_metrics.get("min_latency", 0) * 1000,
                max_latency=latency_metrics.get("max_latency", 0) * 1000,
                ai_interrupting_user=metrics.get("ai_interrupting_user", False),
                user_interrupting_ai=metrics.get("user_interrupting_ai", False),
                ai_user_overlap_count=ai_user_overlap_count,
                user_ai_overlap_count=user_ai_overlap_count,
                talk_ratio=metrics.get("talk_ratio", 0),
                average_pitch=metrics.get("average_pitch", 0),
                words_per_minute=metrics.get("words_per_minute", 0),
            )
            
            logger.info(f"Successfully processed: {path}")
            results.append(result)

        except Exception as e:
            error_msg = f"Error processing file {path}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            results.append(MetricsResponse(
                file_path=path,
                filename=os.path.basename(path),
                status="error",
                error_message=error_msg,
            ))

    # Cleanup downloaded files
    for file_path in downloaded_files:
        downloader.cleanup_temp_file(file_path)

    logger.info(f"Batch processing complete, returning {len(results)} results")
    return results


@app.post("/batch-stream")
async def analyze_batch_stream(audio_files: AudioFileList):
    """
    Stream analyze multiple audio files - returns results as they complete
    
    Returns NDJSON (newline-delimited JSON) where each line is a complete result
    for one processed file. Files are processed sequentially but results are
    streamed immediately as each file completes.
    """
    
    async def generate_results():
        """Async generator that yields results as files complete processing"""
        logger.info(f"Starting streaming batch processing for {len(audio_files.file_paths)} files")
        downloaded_files = []  # Track downloaded files for cleanup
        
        for i, path in enumerate(audio_files.file_paths, 1):
            logger.info(f"[{i}/{len(audio_files.file_paths)}] Processing: {path}")
            
            try:
                local_file_path = path
                
                # Handle URL downloads
                if downloader.is_url(path):
                    try:
                        local_file_path = downloader.download(path)
                        downloaded_files.append(local_file_path)
                        logger.info(f"Downloaded to: {local_file_path}")
                    except Exception as e:
                        error_msg = f"Failed to download file from URL: {str(e)}"
                        logger.error(error_msg)
                        error_result = MetricsResponse(
                            file_path=path,
                            filename=os.path.basename(path),
                            status="error",
                            error_message=error_msg,
                        )
                        logger.info(f"[{i}/{len(audio_files.file_paths)}] Error: {path}")
                        yield error_result.model_dump_json() + "\n"
                        continue

                # Check if local file exists
                elif not os.path.exists(local_file_path):
                    error_msg = f"File not found: {path}"
                    logger.error(error_msg)
                    error_result = MetricsResponse(
                        file_path=path,
                        filename=os.path.basename(path),
                        status="error",
                        error_message=error_msg,
                    )
                    logger.info(f"[{i}/{len(audio_files.file_paths)}] Error: {path}")
                    yield error_result.model_dump_json() + "\n"
                    continue

                # Process the file asynchronously
                filename = os.path.basename(local_file_path)
                logger.info(f"Processing file: {filename}")
                
                # Run the processor in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                metrics = await loop.run_in_executor(None, processor.process_file, local_file_path)

                # Extract latency points
                latency_points = []
                if metrics.get("vad_latency_details"):
                    for point in metrics["vad_latency_details"]:
                        if point["interaction_type"] == "user_to_agent":
                            latency_ms = point.get("latency_ms", 0)
                            if latency_ms == 0 and "latency_seconds" in point:
                                latency_ms = point["latency_seconds"] * 1000

                            latency_points.append({
                                "latency_ms": latency_ms,
                                "moment": point["to_turn_start"],
                            })

                # Extract metrics
                latency_metrics = metrics.get("vad_latency_metrics", {})
                overlap_data = metrics.get("overlap_data", {})

                # Count overlaps
                ai_user_overlap_count = 0
                user_ai_overlap_count = 0
                overlaps = overlap_data.get("overlaps", [])
                for overlap in overlaps:
                    if overlap.get("interrupter") == "ai_agent":
                        ai_user_overlap_count += 1
                    elif overlap.get("interrupter") == "user":
                        user_ai_overlap_count += 1

                # Create successful response
                result = MetricsResponse(
                    file_path=path,
                    filename=filename,
                    status="success",
                    latency_points=latency_points,
                    average_latency=latency_metrics.get("avg_latency", 0) * 1000,
                    p50_latency=latency_metrics.get("p50_latency", 0) * 1000,
                    p90_latency=latency_metrics.get("p90_latency", 0) * 1000,
                    min_latency=latency_metrics.get("min_latency", 0) * 1000,
                    max_latency=latency_metrics.get("max_latency", 0) * 1000,
                    ai_interrupting_user=metrics.get("ai_interrupting_user", False),
                    user_interrupting_ai=metrics.get("user_interrupting_ai", False),
                    ai_user_overlap_count=ai_user_overlap_count,
                    user_ai_overlap_count=user_ai_overlap_count,
                    talk_ratio=metrics.get("talk_ratio", 0),
                    average_pitch=metrics.get("average_pitch", 0),
                    words_per_minute=metrics.get("words_per_minute", 0),
                )

                logger.info(f"[{i}/{len(audio_files.file_paths)}] Completed: {path}")
                yield result.model_dump_json() + "\n"

            except Exception as e:
                error_msg = f"Error processing file {path}: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")

                error_result = MetricsResponse(
                    file_path=path,
                    filename=os.path.basename(path),
                    status="error",
                    error_message=error_msg,
                )
                logger.info(f"[{i}/{len(audio_files.file_paths)}] Error: {path}")
                yield error_result.model_dump_json() + "\n"

        # Cleanup downloaded files
        for file_path in downloaded_files:
            downloader.cleanup_temp_file(file_path)

        logger.info(f"Streaming batch processing complete for {len(audio_files.file_paths)} files")

    return StreamingResponse(
        generate_results(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
