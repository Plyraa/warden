"""
Headless Warden - Pure FastAPI Audio Processing Service
No database, no ElevenLabs, no web UI - just streaming audio analysis
"""
import os
import sys
import uvicorn
from typing import List
from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse
from schemas import AudioFileList, BatchMetricsResponse
from service import VoiceAgentEvaluatorService

sys.path.append(os.getcwd() + "/..")
from data_utils.authenticator import requestAuthenticator
from data_utils.logger import init_logging

service: VoiceAgentEvaluatorService = VoiceAgentEvaluatorService()

# Initialize logger
file_path = os.path.dirname(os.path.realpath(__file__))
logger = init_logging(file_path)

# Initialize FastApi
logger.info("Initializing App")

app = FastAPI(
    title="Voice Agent Evaluation Service",
    description="Headless audio analysis service with streaming batch processing",
    version="1.0"
)

@app.get("/isAlive")
async def is_alive():
    """isAlive() Check - Model monitoring cron job will send get request this endpoint
    every 5 minutes and if response is no;
    related processes (child and parents) will kill on machine.
    ### Return:
    - **status** : 'ok' if api is alive
    """
    return ORJSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/batch", response_model=BatchMetricsResponse)
async def get_batch_metrics(audio_files: AudioFileList, authenticated: bool = Depends(requestAuthenticator)):
    """
    Batch analyze multiple audio files
    
    Takes a list of file paths or URLs and returns metrics for each file.
    Returns partial results even if some files fail.
    """
    return await service.analyze_batch(audio_files)

@app.post("/batch-stream")
async def get_batch_metrics_stream(audio_files: AudioFileList, authenticated: bool = Depends(requestAuthenticator)):
    """
    Stream analyze multiple audio files - returns results as they complete
    
    Returns NDJSON (newline-delimited JSON) where each line is a complete result
    for one processed file. Files are processed sequentially but results are
    streamed immediately as each file completes.
    """
    return await service.analyze_batch_strem(audio_files)

if __name__ == "__main__":
    uvicorn.run(app, port=8030, host= "0.0.0.0")
