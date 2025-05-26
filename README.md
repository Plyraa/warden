# Audio Analysis for Jotform AI Agent Calls

# Warden Architecture and Integration Guide

## Architecture Overview

Warden is a comprehensive audio analysis system designed for evaluating AI agent call quality. It consists of two main components:

1. **FastAPI Server** - Backend service that handles batch processing of audio files and exposes RESTful API endpoints
2. **Flask Web UI** - Frontend interface for interactive analysis and visualization of audio data

### System Architecture

```
                    ┌─────────────────────┐
                    │                     │
                    │  FastAPI Server     │ ◄─── REST API (Port 8000)
                    │  - Audio Analysis   │      - Batch processing
                    │  - Metrics          │      - JSON input/output
                    │  - Database         │
                    │                     │
                    └─────────┬───────────┘
                              │
                              │ Shared components
                              │ - AudioMetricsCalculator
                              │ - Database
                              ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│  Flask Web UI (Port 5000)                       │
│  - Interactive visualization                    │
│  - Audio file selection                         │
│  - Metrics display                              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Core Components

- **AudioMetricsCalculator**: Processes audio files to extract metrics like latency, overlap, and talk ratio
- **Database**: SQLite storage for analyzed audio metrics
- **Visualization**: Generates visual representations of audio analysis

## Integration Guide for Third-Party Systems

### Using the FastAPI Endpoint

The FastAPI server provides a simple REST API for batch processing audio files. To integrate with Warden:

#### Endpoint Details:
- **URL**: `http://127.0.0.1:8000/batch`
- **Method**: POST
- **Content-Type**: application/json

#### Request Format:
```json
{
  "file_paths": [
    "/path/to/local/file.mp3",
    "https://example.com/audio/file.mp3"
  ]
}
```

#### Input Options:
- **Local files**: Provide absolute paths to MP3 files on the server
- **Remote files**: Provide URLs to MP3 files (will be downloaded automatically)

#### Response Format:
```json
[
  {
    "file_path": "/path/to/file.mp3",
    "filename": "file.mp3",
    "latency_points": [
      {"moment": "AI response", "time": 1.24},
      {"moment": "User response", "time": 0.89}
    ],
    "average_latency": 1.07,
    "p50_latency": 1.07,
    "p90_latency": 1.22,
    "min_latency": 0.89,
    "max_latency": 1.24,
    "ai_interrupting_user": false,
    "user_interrupting_ai": false,
    "ai_user_overlap_count": 0,
    "user_ai_overlap_count": 0,
    "talk_ratio": 0.65,
    "average_pitch": 142.3,
    "words_per_minute": 128.5
  }
]
```

### Sample Integration Code

```python
import requests
import json

def analyze_audio_files(file_paths):
    """
    Send audio files for analysis using Warden API
    
    Args:
        file_paths: List of local file paths or URLs
        
    Returns:
        List of analysis results or None if request failed
    """
    try:
        # Prepare the request
        url = "http://127.0.0.1:8000/batch"
        payload = {"file_paths": file_paths}
        
        # Send the request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Return the analysis results
        return response.json()
        
    except Exception as e:
        print(f"Error analyzing audio files: {str(e)}")
        return None

# Example usage
results = analyze_audio_files([
    "C:/path/to/call_recording.mp3", 
    "https://example.com/call_recording.mp3"
])
```

This project provides tools for analyzing audio recordings of Jotform AI Agent calls. It can process audio files to extract various metrics and offers both a web interface for visualization and API endpoints for automated analysis.

## Requirements

- Python 3.12
- Pip

## Installation

1.  Clone the repository or download the project files.
2.  Navigate to the project directory in your terminal.
3.  Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Server

Warden now offers multiple ways to run the application:

1. **Default Mode - FastAPI Server**:
   ```bash
   python .\warden.py
   ```
   This starts only the FastAPI server at `http://127.0.0.1:8000` for batch processing.

2. **Full Mode - FastAPI + Web UI**:
   ```bash
   python .\warden.py --gui
   ```
   This starts both the FastAPI server at `http://127.0.0.1:8000` and the Web UI at `http://127.0.0.1:5000`.

3. **Legacy Web-only Mode**:
   ```bash
   python .\warden.py --web
   ```
   This starts only the web interface at `http://127.0.0.1:5000` (for backward compatibility).
   
4. **Processing Mode**:
   ```bash
   python .\warden.py --process
   ```
   This processes all audio files in the input directory without starting any servers.

You can also use the provided batch script for convenience:
```bash
.\start_api_server.bat
```

#### API Endpoints

The FastAPI server provides the following endpoints:

1. **Web Interface**: `http://127.0.0.1:5000`
   - The original web interface for interactive analysis

2. **Batch Analysis**: `POST http://127.0.0.1:8000/batch`
   - Analyze multiple audio files in one request
   - Request format: JSON with file paths or URLs
   ```json
   {
     "file_paths": [
       "path/to/file1.mp3",
       "path/to/file2.mp3",
       "https://example.com/path/to/file3.mp3"
     ]
   }
   ```
   - Both local file paths and URLs are supported
   - URLs will be downloaded automatically by the server
   - Returns detailed metrics for each file, including:
     - Turn-taking latency points (moment and time)
     - Average, median (P50), P90, min, max latency
     - AI and user interruption info
     - Overlap counts
     - Talk ratio
     - Average pitch
     - Words per minute

#### Example Usage

Use the included example client to test the batch API:
```bash
python .\batch_client_example.py
```

This example demonstrates how to:
1. Analyze local audio files 
2. Analyze audio files from URLs
3. Mix both local files and URLs in a single request

See `batch_client_example.py` for the full source code.

### Command-Line Processing

To process audio files directly from the command line:

```bash
python .\warden.py --input-dir <your_input_directory> --output-dir <your_output_directory>
```

Replace `<your_input_directory>` with the path to the directory containing your audio files (e.g., `stereo_test_calls`) and `<your_output_directory>` with the path where processed files should be saved (e.g., `sampled_test_calls`).

If you omit `--input-dir` and `--output-dir`, it will default to `stereo_test_calls` and `sampled_test_calls` respectively.

For example:

```bash
python .\warden.py
```

This will process files from `stereo_test_calls` and save results to `sampled_test_calls`.

### Available Command-Line Arguments

*   `--input-dir`: Directory containing input audio files (default: `stereo_test_calls`).
*   `--output-dir`: Directory to save processed audio files (default: `sampled_test_calls`).
*   `--web`: Start the web interface for visualization.
*   `--host`: Host address for the web interface (default: `127.0.0.1`).
*   `--port`: Port for the web interface (default: `5000`).

## Features

### Audio Analysis
The system analyzes stereo audio files, where:
- Left channel contains the user's voice
- Right channel contains the AI agent's voice

It extracts the following metrics:
- Response latency (average, P10, P50, P90)
- Speech overlap detection (AI interrupting user, user interrupting AI)
- Talk ratio (agent speaking time / user speaking time)
- Average pitch of the AI agent
- Words per minute for the AI agent

### Transcription
The system now includes advanced transcription capabilities:
- Speech-to-text conversion using ElevenLabs Scribe API
- Word-level timing for precise analysis
- Speaker diarization (separating user and AI agent speech)
- **NEW: Speech overlap detection and highlighting**
- **NEW: Enhanced visualization of speech patterns**

### Speech Overlap Analysis
The new overlap detection feature identifies moments when:
- The AI agent starts speaking while the user is still talking
- The user starts speaking while the AI agent is still talking

This data is visualized in several ways:
1. Word-level timeline with color-coded speakers
2. Highlighted overlap regions in red
3. Statistical analysis of overlap frequency
4. Transcript with overlapping words highlighted

## Setting Up Transcription

To enable transcription functionality:

1. Create a `.env` file in the project root directory
2. Add your ElevenLabs API key:

```
ELEVENLABS_API_KEY=your_api_key_here
```

3. Restart the application

Without an API key, the system will still analyze audio but won't generate transcripts.

## Database Updates

The system now stores additional data about speech overlaps:
- Overlap detection flags
- Word-level timing information
- Overlap counts and statistics

To update your existing database to the new schema:

```bash
python update_database.py
```

**Warning:** This will reset your database and require reprocessing audio files.

## Testing the Transcription Pipeline

To test the transcription and overlap detection functionality:

```bash
python test_elevenlabs_transcript.py
```

This will verify that:
1. The ElevenLabs API response is processed correctly
2. Overlapping speech is detected properly
3. Transcript data is stored correctly in the database

## Optimizing ElevenLabs API Usage

The system now includes an optimization to save API credits:

- When a file has been analyzed before but doesn't have transcript data, the system checks if speech segments (user_windows and agent_windows) are available in the database.
- If speech segments exist, it generates a transcript without calling the ElevenLabs API.
- This script shows the timing and flow of the conversation without actual transcribed words.
- The web UI clearly indicates when a transcript is being displayed.

### When API Calls Are Made

The system will only call the ElevenLabs API when:
1. Processing a new audio file for the first time
2. Manually requesting transcription for a file that has no speech segments in the database

## Using the Batch API Client Example

The project includes `batch_client_example.py`, a demonstration of how to use the Warden batch API endpoint programmatically. This example shows how to:

1. Process multiple local audio files
2. Process audio files from URLs
3. Combine both local files and remote URLs in a single request

### How to Run the Batch Client Example

Run the script using Python:

```bash
python batch_client_example.py
```

The script will:
1. Prompt you to choose between local files, URLs, or both
2. For local files: process files from the "stereo_test_calls" directory
3. For URLs: process MP3 files from the provided URLs
4. Send the combined list to the Warden batch API endpoint
5. Display detailed metrics for each processed file

### Batch Client Example Output

The script displays comprehensive metrics for each processed file:

- **Basic Info**: File name and path
- **Latency Metrics**: Average, P50/P90, Min/Max latency
- **Interruption Detection**: Whether AI or user interrupted each other
- **Overlap Counts**: Number of AI→User and User→AI overlaps
- **Other Metrics**: Talk ratio, average pitch, words per minute
- **Latency Points**: First 5 latency points with timestamp and value

### Adapting the Batch Client Example

You can modify the example script for your specific needs:

1. **Change the server URL**: If you're running the server on a different host or port
2. **Add more URLs**: Include additional URLs in the `mp3_urls` list
3. **Process different local files**: Modify the directory path in `base_dir`
4. **Custom error handling**: Enhance the exception handling for your use case
5. **Output formatting**: Change how results are displayed or save them to a file

## Integrating with Other Systems

The Warden batch API is designed to be easily integrated with other systems. You can:

1. Use the API from any programming language that can make HTTP requests
2. Process large collections of audio files in batch mode
3. Store results in your own database or analytics system
4. Generate custom reports based on the returned metrics
