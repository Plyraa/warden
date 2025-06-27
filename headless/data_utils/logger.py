"""
Mock logger implementation for production deployment
Replaces external data_utils.logger dependency with simple print-based logging
"""
import datetime
from typing import Any


class MockLogger:
    """Simple logger that prints to console with timestamps"""
    
    def __init__(self, name: str = "headless_warden"):
        self.name = name
    
    def _log(self, level: str, message: Any) -> None:
        """Internal method to format and print log messages"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level} - {self.name}: {message}")
    
    def info(self, message: Any) -> None:
        """Log info level message"""
        self._log("INFO", message)
    
    def error(self, message: Any) -> None:
        """Log error level message"""
        self._log("ERROR", message)
    
    def warning(self, message: Any) -> None:
        """Log warning level message"""
        self._log("WARNING", message)
    
    def debug(self, message: Any) -> None:
        """Log debug level message"""
        self._log("DEBUG", message)


def init_logging(file_path: str = None) -> MockLogger:
    """
    Mock implementation of init_logging function
    Returns a MockLogger instance that mimics the original logger behavior
    
    Args:
        file_path: Path to the file (ignored in mock implementation)
        
    Returns:
        MockLogger instance with info() and error() methods
    """
    logger_name = "headless_warden"
    if file_path:
        # Extract a simple name from the file path for logging context
        import os
        logger_name = f"headless_warden.{os.path.basename(file_path)}"
    
    return MockLogger(logger_name)
