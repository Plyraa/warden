"""
Configuration management for Warden Headless
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings"""
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Audio processing settings
    SAMPLE_RATE = 16000  # Required for Silero VAD
    MAX_FILE_SIZE_MB = 500
    
    # File management
    TEMP_DIR = Path(__file__).parent / "temp_downloads"
    CLEANUP_TEMP_FILES = True
    
    @classmethod
    def get_max_file_size_bytes(cls):
        """Get max file size in bytes"""
        return cls.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @classmethod
    def ensure_temp_dir(cls):
        """Ensure temp directory exists"""
        cls.TEMP_DIR.mkdir(exist_ok=True)
        return cls.TEMP_DIR
