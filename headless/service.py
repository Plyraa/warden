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
from llm_evaluator import LlmEvaluator, LlmEvaluationResult

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
        
        # Initialize LLM evaluator with error handling
        try:
            self.llm_evaluator = LlmEvaluator()
            logger.info("✅ LLM Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Evaluator: {e}")
            self.llm_evaluator = None

    async def analyze_batch(self, audio_files: AudioFileList):
        print(audio_files)
        results = []
        downloaded_files: list[str] = []  # Track any files pulled from remote URLs
        logger.info(f"Received batch request for {len(audio_files.files)} files")

        loop = asyncio.get_event_loop()

        for audio_file in audio_files.files:
            path = audio_file.path
            agent_id = audio_file.agent_id
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
                
                metrics = await loop.run_in_executor(None, self.processor.process_file, local_file_path)

                # Run LLM evaluation
                llm_evaluation = None
                if self.llm_evaluator is None:
                    logger.warning(f"LLM Evaluator not available, skipping evaluation for {filename}")
                else:
                    try:
                        logger.info(f"Starting LLM evaluation for {filename} with agent_id {agent_id}")
                        llm_evaluation = await loop.run_in_executor(None, self.llm_evaluator.run_evaluation, local_file_path, agent_id)
                        logger.info(f"LLM evaluation completed successfully for {filename}")
                    except Exception as e:
                        logger.error(f"LLM evaluation failed for {path}: {e}")
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        # Continue processing without LLM evaluation
                
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
                                "moment": point["from_turn_end"]  # Fixed: Mark start of latency period
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
                    toneAdherence=llm_evaluation.toneAdherence if llm_evaluation else None,
                    personaAdherence=llm_evaluation.personaAdherence if llm_evaluation else None,
                    languageSwitch=llm_evaluation.languageSwitch if llm_evaluation else None,
                    sentiment=llm_evaluation.sentiment if llm_evaluation else None,
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
            logger.info(f"Starting streaming batch processing for {len(audio_files.files)} files")
            downloaded_files: list[str] = []  # Track downloaded files for cleanup
            
            for i, audio_file in enumerate(audio_files.files, 1):
                path = audio_file.path
                agent_id = audio_file.agent_id
                logger.info(f"[{i}/{len(audio_files.files)}] Processing: {path}")
                
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
                            logger.info(f"[{i}/{len(audio_files.files)}] Error: {path}")
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
                        logger.info(f"[{i}/{len(audio_files.files)}] Error: {path}")
                        yield error_result.model_dump_json() + "\n"
                        continue

                    # Process the file asynchronously
                    filename = os.path.basename(local_file_path)
                    logger.info(f"Processing file: {filename}")
                    
                    # Run the processor in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    metrics = await loop.run_in_executor(None, self.processor.process_file, local_file_path)

                    # Run LLM evaluation
                    llm_evaluation = None
                    if self.llm_evaluator is None:
                        logger.warning(f"LLM Evaluator not available, skipping evaluation for {filename}")
                    else:
                        try:
                            logger.info(f"Starting LLM evaluation for {filename} with agent_id {agent_id}")
                            llm_evaluation = await loop.run_in_executor(None, self.llm_evaluator.run_evaluation, local_file_path, agent_id)
                            logger.info(f"LLM evaluation completed successfully for {filename}")
                        except Exception as e:
                            logger.error(f"LLM evaluation failed for {path}: {e}")
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            # Continue processing without LLM evaluation

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
                                    "moment": point["from_turn_end"],  # Fixed: Mark start of latency period
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
                        toneAdherence=llm_evaluation.toneAdherence if llm_evaluation else None,
                        personaAdherence=llm_evaluation.personaAdherence if llm_evaluation else None,
                        languageSwitch=llm_evaluation.languageSwitch if llm_evaluation else None,
                        sentiment=llm_evaluation.sentiment if llm_evaluation else None,
                    )

                    logger.info(f"[{i}/{len(audio_files.files)}] Completed: {path}")
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
                    logger.info(f"[{i}/{len(audio_files.files)}] Error: {path}")
                    yield error_result.model_dump_json() + "\n"

            self.downloader.cleanup_temp_dir()
            logger.info(f"Streaming batch processing complete for {len(audio_files.files)} files")

        return StreamingResponse(
            generate_results(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )
