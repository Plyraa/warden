# Warden Headless - Advanced Audio Analysis API

A production-ready FastAPI service for comprehensive audio analysis combining traditional audio metrics with AI-powered semantic evaluation. Features unified Gemini-based analysis for language detection, sentiment analysis, behavioral patterns, and task completion assessment.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PROXY_API_KEY="your-gemini-proxy-api-key"
export GEMINI_PROXY_BASE_URL="your-gemini-proxy-url"

# Start the server
python app.py
# Server runs on http://localhost:8030
```

## 📋 Features

### 🎵 **Audio Metrics**
- **Latency Analysis**: Response time measurements and percentiles
- **Overlap Detection**: AI/user interruptions and talk ratios
- **Audio Quality**: Noise and echo detection
- **Speaking Patterns**: Pitch analysis and words per minute

### 🧠 **AI-Powered Semantic Analysis**
- **Language Switch Detection**: Identifies when agents switch languages
- **Sentiment Analysis**: Categorizes user sentiment (happy, neutral, angry, disappointed)
- **Churn Risk Assessment**: Detects customers at risk of leaving (strict criteria)
- **Repetition Analysis**: Identifies problematic repetitive patterns
- **Task Completion**: Evaluates whether user goals were achieved

### 🔧 **Production Features**
- **Streaming Batch Processing**: Real-time results via `/batch-stream`
- **URL Support**: Download and process remote audio files
- **Unified Analysis**: Single API call for complete evaluation
- **Zero Database Dependencies**: Completely self-contained

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```

### Batch Processing (Complete Analysis)
```bash
POST /batch
Content-Type: application/json

{
  "files": [
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
  "files": ["file1.mp3", "file2.mp3"]
}
```

**Response**: NDJSON stream (one JSON result per line)

**Note**: All endpoints now provide complete analysis including audio metrics + AI semantic evaluation.

## 📊 Response Format

Each processed file returns comprehensive analysis:

```json
{
  "file_path": "/path/to/audio.mp3",
  "filename": "audio.mp3",
  "status": "success",
  "error_message": null,
  
  // Audio Metrics
  "latency_points": [
    {
      "latency_ms": 1250.5,
      "moment": 5.2
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
  "echoInterrupt": false,
  
  // AI Semantic Analysis
  "languageSwitch": false,
  "sentiment": "happy",
  "userChurnRisk": false,
  "userChurnReasoning": null,
  "userRepetition": false,
  "agentRepetition": false,
  "taskCompletion": "Fully Completed",
  "taskCompletionReasoning": "User successfully completed password reset process"
}
```

### Field Descriptions

#### Audio Metrics
| Field | Description |
|-------|-------------|
| `file_path` | Original file path/URL provided |
| `filename` | Extracted filename |
| `status` | "success" or "error" |
| `error_message` | Error details (null on success) |
| `latency_points` | Individual latency measurements with timestamps |
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

#### AI Semantic Analysis
| Field | Description |
|-------|-------------|
| `languageSwitch` | Boolean: Agent switched languages during conversation |
| `sentiment` | User sentiment: "happy", "neutral", "angry", "disappointed" |
| `userChurnRisk` | Boolean: Customer shows explicit intent to leave (strict criteria) |
| `userChurnReasoning` | String: Explanation when churn risk detected (null otherwise) |
| `userRepetition` | Boolean: User repeated requests 3+ times due to agent failure |
| `agentRepetition` | Boolean: Agent showed 3+ instances of identical unhelpful responses |
| `taskCompletion` | Enum: "Fully Completed", "Partially Completed", "Not Completed" |
| `taskCompletionReasoning` | String: One-sentence justification for completion assessment |

## 🛠 Configuration

### Environment Variables
```bash
# Required: Gemini API Configuration
export PROXY_API_KEY="your-gemini-proxy-api-key"
export GEMINI_PROXY_BASE_URL="your-gemini-proxy-base-url"

# Optional: Server Configuration  
export HOST="0.0.0.0"
export PORT="8030"
```

### Application Settings
Edit configuration in the code as needed:

```python
class Config:
    HOST = "0.0.0.0"           # Server host
    PORT = 8030                # Server port  
    SAMPLE_RATE = 16000        # Audio processing sample rate
    MAX_FILE_SIZE_MB = 500     # Maximum file size limit
    TEMP_DIR = Path("temp_downloads")  # Temporary file storage
    CLEANUP_TEMP_FILES = True  # Auto-cleanup downloaded files
```

## 🧠 AI Analysis Details

### Churn Risk Detection (Strict Criteria)
- Requires **BOTH** severe dissatisfaction **AND** explicit intent to leave
- Customer must use harsh language ("terrible", "awful", "useless", etc.)
- Customer must explicitly state intent to stop using service or switch competitors
- Technical issues or workflow problems alone do **NOT** indicate churn risk

### Repetition Detection (3+ Instance Threshold)
- **User Repetition**: User repeats reasonable requests 3+ times because agent ignores them
- **Agent Repetition**: Agent gives identical unhelpful responses 3+ times to different inputs
- Excludes workflow-required repetitions and confirmation requests

### Task Completion Assessment
- **Fully Completed**: User achieved their primary goal
- **Partially Completed**: Some progress made but goal not fully achieved  
- **Not Completed**: User's primary goal was not achieved

### Sentiment Analysis
- **Happy**: Satisfaction, gratitude, positive feedback, successful resolution
- **Neutral**: Factual exchanges, no strong emotional indicators
- **Angry**: Frustration, harsh language, complaints, aggressive tone
- **Disappointed**: Unmet expectations, mild frustration, resigned acceptance
## 🚀 Production Deployment

### Standard Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Set required environment variables
export PROXY_API_KEY="your-gemini-proxy-api-key"
export GEMINI_PROXY_BASE_URL="your-gemini-proxy-url"

# Run the server
python app.py
```

### Batch Processing Script
For local file processing without the web server:

```bash
# Edit fast_llm_logger.py configuration
# Set AUDIO_FILES_DIR and INPUT_CSV paths
python fast_llm_logger.py
```

## 📁 File Input Options

1. **Relative paths**: `"audio.mp3"` (looks in configured directory)
2. **Absolute paths**: `"/full/path/to/audio.mp3"`
3. **URLs**: `"https://example.com/audio.mp3"`

Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC

## 📋 Requirements

- Python 3.8+
- Gemini API access via proxy
- PyTorch (for Silero VAD)
- librosa (audio processing)
- FastAPI + uvicorn (web server)
- See `requirements.txt` for complete list

## 🔧 Architecture

```
headless/
├── app.py                    # Main FastAPI application
├── service.py               # Core service orchestration
├── llm_evaluator.py         # Unified Gemini-based semantic analysis
├── audio_processor.py       # Traditional audio metrics
├── url_downloader.py        # URL handling and downloads
├── fast_llm_logger.py       # Batch processing script
├── prompts.yaml            # AI analysis prompts and templates
├── schemas.py              # Data models and response schemas
└── requirements.txt        # Dependencies
```

## 🎯 Analysis Pipeline

1. **Audio Preprocessing**: Download URLs, validate formats, prepare for analysis
2. **Audio Metrics**: Extract latency, overlaps, talk ratios, noise/echo detection
3. **Semantic Analysis**: Unified Gemini call for language, sentiment, behavioral analysis
4. **Result Combination**: Merge audio metrics with AI insights
5. **Response**: Return comprehensive analysis with structured data

## 🚨 Error Handling

- Graceful degradation for individual file failures
- Detailed error messages in responses
- Automatic cleanup of temporary files
- Request validation and file format checking
- Fallback values for failed AI analysis
- Structured error reporting for debugging

## 📈 Performance

- **Unified Analysis**: Single API call combines audio + AI evaluation
- **Async Processing**: Non-blocking operations for better throughput
- **Streaming Responses**: Immediate feedback for batch operations
- **Memory Efficient**: Optimized audio processing pipeline
- **Resource Management**: Configurable limits and cleanup

## 🔒 Security & Best Practices

- **API Key Management**: Secure environment variable configuration
- **URL Validation**: Safe download verification for remote files
- **Input Sanitization**: Comprehensive validation and format checking
- **Temporary File Cleanup**: Automatic removal of downloaded content
- **Rate Limiting**: Respectful API usage with built-in delays
- **Error Boundaries**: Isolated failure handling per file

## 💡 Use Cases

- **Customer Service Quality**: Analyze support call effectiveness
- **Agent Performance**: Identify language switches and repetitive responses  
- **User Experience**: Track sentiment and task completion rates
- **Churn Prevention**: Early detection of at-risk customers
- **Training Data**: Generate insights for agent improvement
- **Quality Assurance**: Automated conversation analysis at scale

---

**Note**: This system provides comprehensive analysis combining traditional audio metrics with advanced AI semantic evaluation. The unified approach reduces complexity while providing deeper insights into conversation quality and user experience.
