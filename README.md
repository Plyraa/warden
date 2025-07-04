# Warden - AI Agent Call Audio Analysis

A comprehensive audio analysis system for evaluating AI agent call quality with real-time streaming capabilities and advanced LLM-powered evaluation features.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# ELEVENLABS_API_KEY=your_elevenlabs_key
# OPENAI_API_KEY=your_openai_key

# Start both API server (8000) and Web UI (5000)
python warden.py

# API-only mode
python warden.py --api-only

# Web-only mode  
python warden.py --web-only
```

## Core Features

- **Real-time Streaming**: Process multiple files with live results via `/batch-stream`
- **Batch Processing**: Analyze multiple files at once via `/batch`
- **Audio Metrics**: Latency, overlap detection, talk ratio, pitch, WPM
- **Transcription**: Speech-to-text with overlap analysis (ElevenLabs API)
- **LLM Evaluation** (NEW): AI agent persona adherence, language consistency, and user sentiment analysis
- **Noise Reduction** (NEW): Optional Facebook Denoiser for cleaner user audio
- **Web UI**: Interactive visualization and file management with agent ID input
- **URL Support**: Process remote audio files automatically

## NEW Features 🚀

### LLM-Powered Agent Evaluation
- **Persona Adherence**: 1-5 scale rating of how well the agent maintains its defined character
- **Language Switch Detection**: Identifies if the agent switched languages unexpectedly  
- **User Sentiment Analysis**: Categorizes user emotion as happy/neutral/angry/disappointed
- **Agent Properties Integration**: Fetches agent configuration from JotForm API

### Noise Reduction
- **Facebook Denoiser**: Optional noise reduction for user audio channel
- **UI Toggle**: Enable/disable noise reduction in the web interface
- **Channel-Specific**: Only applies to user channel, preserving agent audio quality

## API Endpoints

### 🚀 Streaming Batch Processing (NEW)
**`POST /batch-stream`** - Get results as files complete processing

```bash
curl -X POST "http://localhost:8000/batch-stream" \
  -H "Content-Type: application/json" \
  --no-buffer \
  -d '{"file_paths": ["file1.mp3", "file2.mp3"]}'
```

**Response**: NDJSON stream (one JSON result per line)
```json
{"filename": "file1.mp3", "status": "success", "average_latency": 1250.5, ...}
{"filename": "file2.mp3", "status": "success", "average_latency": 980.2, ...}
```

### 📊 Standard Batch Processing
**`POST /batch`** - Wait for all files to complete

```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["file1.mp3", "file2.mp3"]}'
```

**Response**: JSON array with all results

### Input Options
- **Local files**: `"test1.mp3"` (looks in `stereo_test_calls/`)
- **Absolute paths**: `"/full/path/to/file.mp3"`
- **URLs**: `"https://example.com/audio.mp3"`

## Response Format

```json
{
  "file_path": "C:\\Users\\...\\stereo_test_calls\\243801406824559add7684.37683750.mp3",
  "filename": "243801406824559add7684.37683750.mp3", 
  "status": "success",
  "error_message": null,
  "latency_points": [
    {"latency_ms": 1000.0, "moment": 5.2},
    {"latency_ms": 3099.9999999999977, "moment": 20.9}
  ],
  "average_latency": 2400.000000000002,
  "p50_latency": 2500.0,
  "p90_latency": 3899.9999999999914,
  "min_latency": 200.00000000001705,
  "max_latency": 4099.9999999999945,
  "ai_interrupting_user": false,
  "user_interrupting_ai": true,
  "ai_user_overlap_count": 0,
  "user_ai_overlap_count": 4,
  "talk_ratio": 4.734426229508204,
  "average_pitch": 320.99127197265625,
  "words_per_minute": 206.37119113573402
}
```

**Field Descriptions:**
- `file_path`: Full absolute path to the processed audio file
- `filename`: Just the filename portion
- `status`: "success" or "error"
- `error_message`: null on success, error description on failure
- `latency_points`: List of detected latency measurements
  - `latency_ms`: Response latency in milliseconds
  - `moment`: Time position in the audio file (seconds)
- `average_latency`: Mean latency across all measurements (ms)
- `p50_latency`/`p90_latency`: 50th/90th percentile latencies (ms)
- `min_latency`/`max_latency`: Minimum/maximum detected latencies (ms)
- `ai_interrupting_user`/`user_interrupting_ai`: Boolean overlap detection
- `ai_user_overlap_count`/`user_ai_overlap_count`: Number of interruptions
- `talk_ratio`: Ratio of user speech time to AI speech time
- `average_pitch`: Mean pitch frequency (Hz)
- `words_per_minute`: Speech rate calculation

## Project Structure

### Core Files
- **`warden.py`** - Main entry point, orchestrates API and Web UI servers
- **`audio_metrics.py`** - Core audio analysis engine with latency/overlap detection
- **`fastapi_server.py`** - FastAPI server providing REST endpoints for batch processing
- **`web_app.py`** - Flask web interface for interactive file upload and visualization
- **`database.py`** - SQLite database operations for storing analysis results
- **`url_helper.py`** - URL processing and remote file download utilities
- **`visualization.py`** - Chart generation and data visualization components
- **`update_database.py`** - Database schema migration and update utility

### Directories
- **`templates/`** - HTML templates for the web interface
- **`static/`** - Static assets (images, CSS, JS) for web UI
- **`database/`** - SQLite database files storage
- **`stereo_test_calls/`** - Sample stereo audio files for testing (User=Left, AI=Right)
- **`sampled_test_calls/`** - Downsampled audio files for faster processing

## Audio Requirements

- **Format**: Stereo MP3/WAV files
- **Channels**: Left = User, Right = AI Agent
- **Location**: Place files in `stereo_test_calls/` directory

## Configuration

### Environment Variables
```bash
# Optional: For transcription features
ELEVENLABS_API_KEY=your_api_key_here
```

### Command Line Options
```bash
python warden.py [options]
  --api-only          # Start only API server
  --web-only          # Start only Web UI  
  --host HOST         # Server host (default: 127.0.0.1)
  --api-port PORT     # API port (default: 8000)
  --web-port PORT     # Web port (default: 5000)
  --threads N         # Waitress threads (default: 4)
  --input-dir DIR     # Audio input directory
  --output-dir DIR    # Processed audio output directory
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Flask Web UI  │
│   (Port 8000)   │    │   (Port 5000)   │
│                 │    │   via Waitress  │
│ • /batch        │    │ • Visualization │
│ • /batch-stream │    │ • File Upload   │
│ • /health       │    │ • Interactive   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
           ┌─────────▼──────────┐
           │ AudioMetricsCalc   │
           │ + Database         │
           │ + Transcription    │
           └────────────────────┘
```
