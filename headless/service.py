"""
Headless Warden - Pure FastAPI Audio Processing Service
No database, no ElevenLabs, no web UI - just streaming audio analysis
"""
import os
import sys
import asyncio
import logging
import traceback
from typing import List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from schemas import AudioFileList, MetricsResponse, HealthResponse, BatchMetricsResponse
from audio_processor import AudioProcessor
from url_downloader import URLDownloader

sys.path.append(os.getcwd() + "/..")
from data_utils.logger import init_logging

# Initialize logger
file_path = os.path.dirname(os.path.realpath(__file__))
logger = init_logging(file_path)

class VoiceAgentEvaluatorService:
    """
    Voice Agent Evaluator Service
    @version 1.0
    """

    def __init__(self) -> object:
        # Use a stable directory under the project root so that manual cleanup can be performed
        self.audio_dir = Path("audio_downloads")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AudioProcessor(audio_dir=self.audio_dir)
        self.downloader = URLDownloader(audio_dir=self.audio_dir)

    async def analyze_batch(self, audio_files: AudioFileList):
        print(audio_files)
        results = []
        downloaded_files: list[str] = []  # Track any files pulled from remote URLs
        logger.info(f"Received batch request for {len(audio_files.file_paths)} files")

        for path in audio_files.file_paths:
            logger.info(f"Processing: {path}")
            
            try:
                local_file_path = path
                
                # Handle URL downloads
                if self.downloader.is_url(path):
                    try:
                        local_file_path = self.downloader.download(path)
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
                
                metrics = self.processor.process_file(local_file_path)
                
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

        self.downloader.cleanup_temp_dir()

        logger.info(f"Batch processing complete, returning {len(results)} results")
        return BatchMetricsResponse(results=results)

    async def analyze_batch_strem(self, audio_files: AudioFileList):
        async def generate_results():
            """Async generator that yields results as files complete processing"""
            logger.info(f"Starting streaming batch processing for {len(audio_files.file_paths)} files")
            downloaded_files: list[str] = []  # Track downloaded files for cleanup
            
            for i, path in enumerate(audio_files.file_paths, 1):
                logger.info(f"[{i}/{len(audio_files.file_paths)}] Processing: {path}")
                
                try:
                    local_file_path = path
                    
                    # Handle URL downloads
                    if self.downloader.is_url(path):
                        try:
                            local_file_path = self.downloader.download(path)
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
                    metrics = await loop.run_in_executor(None, self.processor.process_file, local_file_path)

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

            self.downloader.cleanup_temp_dir()
            logger.info(f"Streaming batch processing complete for {len(audio_files.file_paths)} files")

        return StreamingResponse(
            generate_results(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )
