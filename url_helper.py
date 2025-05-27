"""
Helper functions for handling URL operations, like downloading MP3 files
More robust implementation with retry logic, validation and error handling
"""

import os
import time
import traceback
import re
import requests
import subprocess
import hashlib
import logging
import mimetypes
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from typing import Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import HTTPException

# Configure logging
logger = logging.getLogger(__name__)


class AudioDownloadError(Exception):
    """Custom exception for audio download errors"""

    pass


class AudioDownloader:
    """Robust audio file downloader with retry logic and comprehensive validation"""

    def __init__(
        self, download_dir: str = None, max_file_size: int = 500 * 1024 * 1024
    ):  # 500MB default
        self.download_dir = (
            Path(download_dir) if download_dir else Path.cwd() / "stereo_test_calls"
        )
        self.max_file_size = max_file_size
        self.session = self._create_session()

        # Supported audio formats
        self.supported_extensions = {
            ".mp3",
            ".wav",
            ".m4a",
            ".flac",
            ".ogg",
            ".aac",
            ".wma",
        }
        self.supported_mime_types = {
            "audio/mpeg",
            "audio/wav",
            "audio/wave",
            "audio/x-wav",
            "audio/mp4",
            "audio/m4a",
            "audio/flac",
            "audio/ogg",
            "audio/aac",
            "audio/x-ms-wma",
            "application/octet-stream",
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
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        )

        return session

    def is_url(self, path: str) -> bool:
        """Check if a string is a valid URL"""
        try:
            result = urlparse(path)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    def _get_filename_from_url(self, url: str, content_disposition: str = None) -> str:
        """Extract filename from URL or Content-Disposition header"""
        filename = None

        # Try to get filename from Content-Disposition header
        if content_disposition:
            cd_match = re.search(r"filename[*]?=([^;]+)", content_disposition)
            if cd_match:
                filename = cd_match.group(1).strip("\"'")
                filename = unquote(filename)  # URL decode
                # Extract only the basename to avoid path inclusion
                filename = os.path.basename(filename)
                print(
                    f"Raw filename from Content-Disposition: {cd_match.group(1).strip('"\'')}"
                )
                print(f"After processing: {filename}")
            else:
                print("No filename pattern found in Content-Disposition header")
        # Fallback to URL path
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(unquote(parsed_url.path))

            # Try query parameters if path doesn't yield usable filename
            if not filename or not any(c.isalnum() for c in filename):
                query_params = parse_qs(parsed_url.query)
                for param in [
                    "file",
                    "name",
                    "filename",
                    "audio",
                    "source",
                    "src",
                    "path",
                ]:
                    if param in query_params and query_params[param]:
                        param_value = query_params[param][0]
                        if param_value:
                            potential_filename = os.path.basename(param_value)
                            if potential_filename:
                                filename = potential_filename
                                break
            print(f"Extracted filename from URL: {filename}")

        # Clean filename and ensure it has a valid extension
        if filename:
            filename = self._sanitize_filename(filename)
            if not any(
                filename.lower().endswith(ext) for ext in self.supported_extensions
            ):
                filename = f"{filename}.mp3"
        else:
            # Generate unique filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"audio_{int(time.time())}_{url_hash}.mp3"

        return filename

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 200:
            name = name[:200]

        return f"{name}{ext}"

    def _validate_audio_content(
        self, file_path: Path, content_type: str = None
    ) -> bool:
        """Validate that the downloaded file is likely an audio file"""
        try:
            # Check file size
            if file_path.stat().st_size == 0:
                raise AudioDownloadError("Downloaded file is empty")

            # Check MIME type
            guessed_type, _ = mimetypes.guess_type(str(file_path))
            if guessed_type and guessed_type in self.supported_mime_types:
                return True

            if content_type and any(
                mime in content_type.lower() for mime in self.supported_mime_types
            ):
                return True

            # Check file signature (magic bytes)
            with open(file_path, "rb") as f:
                header = f.read(12)

                # Common audio file signatures
                audio_signatures = [
                    b"ID3",  # MP3 with ID3 tag
                    b"\xff\xfb",  # MP3
                    b"\xff\xf3",  # MP3
                    b"\xff\xf2",  # MP3
                    b"RIFF",  # WAV (followed by WAVE)
                    b"fLaC",  # FLAC
                    b"OggS",  # OGG
                    b"\x00\x00\x00\x20ftypM4A",  # M4A
                ]

                for sig in audio_signatures:
                    if header.startswith(sig):
                        return True

                # Additional check for WAV files
                if header.startswith(b"RIFF") and b"WAVE" in header:
                    return True

            # Try ffprobe validation as a last resort
            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(file_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

                if result.returncode == 0 and result.stdout.strip():
                    codec = result.stdout.strip()
                    logger.info(f"Verified audio file with codec: {codec}")
                    return True
            except Exception as ffprobe_err:
                logger.warning(f"ffprobe validation failed: {ffprobe_err}")

            logger.warning(f"Could not validate audio format for {file_path}")
            return True  # Allow file if we can't determine format

        except Exception as e:
            logger.error(f"Error validating audio content: {e}")
            return False

    def _check_file_size_before_download(self, url: str) -> bool:
        """Check file size via HEAD request before downloading"""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            content_length = response.headers.get("Content-Length")

            if content_length:
                size = int(content_length)
                if size > self.max_file_size:
                    raise AudioDownloadError(
                        f"File too large: {size} bytes (max: {self.max_file_size})"
                    )
                logger.info(f"Expected file size: {size} bytes")

            return True
        except requests.RequestException as e:
            logger.warning(f"Could not check file size via HEAD request: {e}")
            return True  # Continue with download

    def download_audio_from_url(self, url: str, custom_filename: str = None) -> str:
        """
        Download an audio file from URL with comprehensive error handling

        Args:
            url: URL of the audio file
            custom_filename: Optional custom filename (will be sanitized)

        Returns:
            Path to the downloaded file

        Raises:
            AudioDownloadError: If download fails or file is invalid
        """
        if not self.is_url(url):
            raise AudioDownloadError(f"Invalid URL: {url}")

        logger.info(f"Starting download from: {url}")
        print(f"Starting download from: {url}")

        try:
            # Check file size before downloading
            self._check_file_size_before_download(url)

            # Create download directory
            self.download_dir.mkdir(parents=True, exist_ok=True)

            # Start download
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Get filename
            content_disposition = response.headers.get("Content-Disposition", "")

            # Pretty print the Content-Disposition header
            # print("\n=== Content-Disposition Header Details ===")
            # print(f"Raw Content-Disposition: '{content_disposition}'")
            # print(f"Headers: {dict(response.headers)}")
            # print("=" * 50)

            if custom_filename:
                filename = self._sanitize_filename(custom_filename)
                if not any(
                    filename.lower().endswith(ext) for ext in self.supported_extensions
                ):
                    filename = f"{filename}.mp3"
            else:
                filename = self._get_filename_from_url(url, content_disposition)

            # Handle filename conflicts
            save_path = self.download_dir / filename
            counter = 1
            original_stem = save_path.stem
            original_suffix = save_path.suffix

            while save_path.exists():
                filename = f"{original_stem}_{counter}{original_suffix}"
                save_path = self.download_dir / filename
                counter += 1

            # Download with progress tracking
            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")

            logger.info(f"Downloading to: {save_path}")
            print(f"Downloading to: {save_path}")
            print(f"Content-Type: {content_type}")

            downloaded_size = 0

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Check size limit during download
                        if downloaded_size > self.max_file_size:
                            save_path.unlink()  # Delete partial file
                            raise AudioDownloadError(
                                f"File too large during download: {downloaded_size} bytes"
                            )

            # Validate downloaded file
            if not self._validate_audio_content(save_path, content_type):
                save_path.unlink()  # Delete invalid file
                raise AudioDownloadError(
                    "Downloaded file does not appear to be a valid audio file"
                )

            file_size = save_path.stat().st_size
            logger.info(f"Successfully downloaded: {save_path} ({file_size} bytes)")
            print(f"Successfully downloaded: {save_path} ({file_size} bytes)")

            return str(save_path)

        except requests.RequestException as e:
            error_message = f"Network error downloading {url}: {str(e)}"
            if hasattr(e, "response") and e.response:
                status_code = e.response.status_code
                error_message += f" (Status code: {status_code})"

                # Add more context for common HTTP errors
                if status_code == 403:
                    error_message += " - Access forbidden. The server may require authentication or block direct downloads."
                elif status_code == 404:
                    error_message += " - File not found on server."
                elif status_code == 429:
                    error_message += " - Too many requests. Server is rate limiting."
                elif status_code >= 500:
                    error_message += " - Server error."

            raise AudioDownloadError(error_message)
        except Exception as e:
            raise AudioDownloadError(f"Unexpected error downloading {url}: {str(e)}")
        finally:
            # Clean up session if needed
            pass


# Create a singleton downloader instance
_downloader = AudioDownloader()


# Compatibility functions for existing API
def is_url(path: str) -> bool:
    """Compatibility function for API - Check if a string is a valid URL"""
    return _downloader.is_url(path)


def download_audio_from_url(url: str, custom_filename: str = None) -> str:
    """Compatibility function for API - Download an audio file from URL"""
    return _downloader.download_audio_from_url(url, custom_filename)
