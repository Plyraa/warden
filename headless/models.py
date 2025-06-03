"""
Pydantic models for API requests and responses
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class AudioFileList(BaseModel):
    """Model for list of audio files to analyze in batch mode"""
    file_paths: List[str]  # Can be local file paths or URLs


class MetricsResponse(BaseModel):
    """Model for the metrics response"""
    file_path: str  # Keep for backward compatibility
    filename: str  # Use filename instead of file_name
    status: str  # "success" or "error"
    error_message: Optional[str] = None  # Only present if status is "error"
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


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
