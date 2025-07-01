"""
Pydantic models for API requests and responses
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class AudioFile(BaseModel):
    path: str
    agent_id: str

class AudioFileList(BaseModel):
    """Model for list of audio files to analyze in batch mode"""
    files: List[AudioFile]

    class Config:
        schema_extra = {
            "example": {
                "files": [
                    {
                        "path": "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/66057328368247d623d0b87.67876133.mp3",
                        "agent_id": "0197362dee337c83853df36020378b3390f8"
                    },
                    {
                        "path": "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/17457745626800ad609b0bd7.58327851.mp3",
                        "agent_id": "0197362dee337c83853df36020378b3390f8"
                    }
                ]
            }
        }


class MetricsResponse(BaseModel):
    """Model for the metrics response"""
    file_path: str
    filename: str
    status: str
    error_message: Optional[str] = None
    latency_points: List[Dict[str, Any]] = []
    average_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    ai_interrupting_user: bool = False
    user_interrupting_ai: bool = False
    ai_user_overlap_count: int = 0
    user_ai_overlap_count: int = 0
    talk_ratio: float = 0.0
    average_pitch: float = 0.0
    words_per_minute: float = 0.0
    # LLM Evaluation Metrics
    personaAdherence: Optional[int] = Field(None, description="Adherence to the specified persona, from 1 to 5.", ge=1, le=5)
    languageSwitch: Optional[bool] = Field(None, description="Whether the agent switched languages.")
    sentiment: Optional[Literal["happy", "neutral", "angry", "disappointed"]] = Field(None, description="The user's sentiment.")

class BatchMetricsResponse(BaseModel):
    """Model for the batch metrics response"""
    results: List[MetricsResponse]

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
