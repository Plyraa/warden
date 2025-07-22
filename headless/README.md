# Warden Headless - Production Audio Analysis API

A lightweight, production-ready FastAPI service for real-time audio analysis with streaming batch processing. No database dependencies, completely self-contained.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# Server runs on http://localhost:8000
```

## 📋 Features

- **Streaming Batch Processing**: Get results as files complete via `/batch-stream`
- **Standard Batch Processing**: Process multiple files via `/batch`
- **URL Support**: Download and process remote audio files
- **Audio Metrics**: Latency, overlap detection, talk ratio, pitch analysis, noise and echo detection
- **Production Ready**: CORS enabled, structured logging, error handling
- **Zero Dependencies**: No database, no external services required

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```

### Batch Processing (Wait for All)
```bash
POST /batch
Content-Type: application/json

{
  "file_paths": [
    "audio1.mp3",
    "/absolute/path/to/audio2.wav", 
    "https://example.com/audio3.mp3"
  ]
}
```

### Streaming Batch Processing (Real-time Results)
```bash
POST /batch-stream
Content-Type: application/json

{
  "file_paths": ["file1.mp3", "file2.mp3"]
}
```

**Response**: NDJSON stream (one JSON result per line)

## 📊 Response Format

Each processed file returns:

```json
{
  "file_path": "/path/to/audio.mp3",
  "filename": "audio.mp3",
  "status": "success",
  "error_message": null,
  "latency_points": [
    {
      "latency_ms": 1250.5,
      "moment": 5.2
    },
        {
      "latency_ms": 3732.5,
      "moment": 18.4
    }
  ],
  "average_latency": 1250.5,
  "p50_latency": 1200.0,
  "p90_latency": 2000.0,
  "min_latency": 800.0,
  "max_latency": 2500.0,
  "ai_interrupting_user": false,
  "user_interrupting_ai": true,
  "ai_user_overlap_count": 0,
  "user_ai_overlap_count": 3,
  "talk_ratio": 2.5,
  "average_pitch": 180.5,
  "words_per_minute": 150.2,
  "hasNoise": false,
  "noiseInterrupt": false,
  "hasEcho": false,
  "echoInterrupt": false
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `file_path` | Original file path/URL provided |
| `filename` | Extracted filename |
| `status` | "success" or "error" |
| `error_message` | Error details (null on success) |
| `latency_points` | Individual latency measurements with timestamps where latency period begins (user ends talking) |
| `average_latency` | Mean response latency (ms) |
| `p50_latency` / `p90_latency` | 50th/90th percentile latencies (ms) |
| `min_latency` / `max_latency` | Minimum/maximum latencies (ms) |
| `ai_interrupting_user` | Boolean: AI interrupted user |
| `user_interrupting_ai` | Boolean: User interrupted AI |
| `ai_user_overlap_count` | Count of AI interruptions |
| `user_ai_overlap_count` | Count of user interruptions |
| `talk_ratio` | Ratio of AI to user speaking time |
| `average_pitch` | Average pitch in Hz |
| `words_per_minute` | Speaking rate calculation |
| `hasNoise` | Boolean: Noise was detected |
| `noiseInterrupt` | Boolean: Noise interrupted the agent |
| `hasEcho` | Boolean: Echo was detected |
| `echoInterrupt` | Boolean: Echo interrupted the agent |

## 🛠 Configuration

Edit `config.py` to customize:

```python
class Config:
    HOST = "0.0.0.0"           # Server host
    PORT = 8000                # Server port
    SAMPLE_RATE = 16000        # Audio processing sample rate
    MAX_FILE_SIZE_MB = 500     # Maximum file size limit
    TEMP_DIR = Path("temp_downloads")  # Temporary file storage
    CLEANUP_TEMP_FILES = True  # Auto-cleanup downloaded files
```
## 🚀 Production Deployment

### Standard Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Or use the built-in startup
python start.py
```

## 📁 File Input Options

1. **Relative paths**: `"audio.mp3"` (looks in parent `stereo_test_calls/`)
2. **Absolute paths**: `"/full/path/to/audio.mp3"`
3. **URLs**: `"https://example.com/audio.mp3"`

Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC

## 📋 Requirements

- Python 3.8+
- PyTorch (for Silero VAD)
- librosa (audio processing)
- FastAPI + uvicorn (web server)
- See `requirements.txt` for complete list

## 🔧 Architecture

```
headless/
├── app.py              # Main FastAPI application
├── audio_processor.py  # Core audio analysis engine
├── url_downloader.py   # URL handling and downloads
├── models.py          # Pydantic request/response models
├── config.py          # Configuration settings
├── requirements.txt   # Dependencies
└── temp_downloads/    # Temporary file storage
```

## 🚨 Error Handling

- Graceful degradation for individual file failures
- Detailed error messages in responses
- Automatic cleanup of temporary files
- Request validation and file format checking

## 📈 Performance

- Async processing for non-blocking operations
- Streaming responses for immediate feedback
- Memory-efficient audio processing
- Configurable resource limits

## 🔒 Security Notes

- URL validation for safe downloads
- Temporary file cleanup
- Input sanitization and validation

---
