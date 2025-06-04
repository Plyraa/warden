"""
URL downloader for headless Warden - simplified without database dependencies
"""
import os
import time
import re
import requests
import hashlib
import logging
from urllib.parse import urlparse
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config

logger = logging.getLogger(__name__)

class AudioDownloadError(Exception):
    """Custom exception for audio download errors"""
    pass

class URLDownloader:
    """Robust audio file downloader with retry logic and comprehensive validation"""
    
    def __init__(self, download_dir: str = None, max_file_size: int = None):
        self.download_dir = Path(download_dir) if download_dir else Config.TEMP_DIR
        self.max_file_size = max_file_size or Config.get_max_file_size_bytes()
        self.session = self._create_session()
        
        # Create temp directory if it doesn't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Supported audio formats
        self.supported_extensions = {
            ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"
        }
        self.supported_mime_types = {
            "audio/mpeg", "audio/wav", "audio/wave", "audio/x-wav",
            "audio/mp4", "audio/m4a", "audio/flac", "audio/ogg",
            "audio/aac", "audio/x-ms-wma", "application/octet-stream"
        }

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and proper headers"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set user agent to avoid bot detection
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        
        return session

    def download(self, url: str) -> str:
        """
        Download audio file from URL and return local file path
        Always overwrites existing files - request is king!
        
        Args:
            url: URL to download from
            
        Returns:
            str: Path to downloaded file
            
        Raises:
            AudioDownloadError: If download fails
        """
        try:
            # Simple filename from URL
            parsed_url = urlparse(url)
            original_name = os.path.basename(parsed_url.path) or "audio"
            
            # Clean filename and ensure audio extension
            original_name = re.sub(r'[^\w\-_\.]', '_', original_name)
            if not original_name.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')):
                original_name += '.mp3'
                
            local_path = self.download_dir / original_name
            
            logger.info(f"Downloading from URL: {url} -> {local_path}")
            
            # Check file size before downloading
            try:
                head_response = self.session.head(url, timeout=30)
                head_response.raise_for_status()
                
                content_length = head_response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    raise AudioDownloadError(f"File too large: {content_length} bytes (max: {self.max_file_size})")
                    
            except requests.exceptions.RequestException:
                # HEAD request failed, continue with GET
                pass
            
            # Download the file (overwrite if exists)
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Download with size checking
            total_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > self.max_file_size:
                            f.close()
                            local_path.unlink()  # Delete partial file
                            raise AudioDownloadError(f"File too large during download: {total_size} bytes")
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded {total_size} bytes to {local_path}")
            return str(local_path)
            
        except requests.exceptions.RequestException as e:
            raise AudioDownloadError(f"Network error downloading {url}: {str(e)}")
        except Exception as e:
            raise AudioDownloadError(f"Unexpected error downloading {url}: {str(e)}")
    
    def is_url(self, path: str) -> bool:
        """Check if a string is a valid URL"""
        try:
            result = urlparse(path)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False


    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary downloaded file"""
        try:
            if os.path.exists(file_path) and str(Config.TEMP_DIR) in file_path:
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")