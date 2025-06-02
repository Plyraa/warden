# Warden - AI Agent Call Audio Analysis

A comprehensive audio analysis system for evaluating AI agent call quality with real-time streaming capabilities.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

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
- **Web UI**: Interactive visualization and file management
- **URL Support**: Process remote audio files automatically

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
  "file_path": "test1.mp3",
  "filename": "test1.mp3", 
  "status": "success",
  "latency_points": [{"moment": "AI response", "latency_ms": 1240}],
  "average_latency": 1150.5,
  "p50_latency": 1100.0,
  "p90_latency": 1800.0,
  "min_latency": 650.0,
  "max_latency": 2100.0,
  "ai_interrupting_user": false,
  "user_interrupting_ai": true,
  "ai_user_overlap_count": 0,
  "user_ai_overlap_count": 2,
  "talk_ratio": 0.75,
  "average_pitch": 142.3,
  "words_per_minute": 128.5
}
```

## Testing

### Streaming Tests
```bash
# Test with curl
tests/test_curl_streaming.bat

# Test with Python  
python tests/test_real_streaming.py
```

### Standard Tests
```bash
# Test batch API
python batch_client_example.py

# Test transcription
python test_elevenlabs_transcript.py
```

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
│                 │    │                 │
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

## Integration Examples

### Python Client
```python
import requests
import json

# Streaming processing
response = requests.post(
    "http://localhost:8000/batch-stream",
    json={"file_paths": ["file1.mp3", "file2.mp3"]},
    stream=True
)

for line in response.iter_lines(decode_unicode=True):
    if line.strip():
        result = json.loads(line)
        print(f"✅ {result['filename']}: {result['average_latency']:.1f}ms")
```