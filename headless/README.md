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

### � **Scripted Initialization Analysis**
- Detects early scripted transitions (attempt2, attempt1, terminate, verification, disclaimer, first_agent_message)
- Supports conversation types: shared-phone, private-phone, web-audio, presentation
- Matches templates from `audio_files/` (prefers MP3, falls back to WAV)
- Merges agent VAD segments with ≤1.5s gaps before matching for robustness
- Produces `initial_latency_points` with interval latencies per stage (seconds)

### �🧠 **AI-Powered Semantic Analysis**
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
GET /isAlive
```

### Batch Processing (Complete Analysis)
```bash
POST /batch
Content-Type: application/json

{
  "files": [
    { "path": "audio1.mp3", "conversation_type": "shared-phone" },
    { "path": "/absolute/path/to/audio2.wav", "conversation_type": "private-phone" },
    { "path": "https://example.com/audio3.mp3", "conversation_type": "web-audio" }
  ]
}
```

Behavioral (LLM) metrics are optional. Enable with a query parameter:

```bash
POST /batch?run_behavioral=true
```

### Streaming Batch Processing (Real-time Results)
```bash
POST /batch-stream
Content-Type: application/json

{
  "files": [
    { "path": "file1.mp3", "conversation_type": "presentation" },
    { "path": "file2.mp3" }
  ]
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
  
  // Scripted Initialization Analysis
  "conversation_type": "shared-phone",
  "initial_latency_points": {
    "attempt2": 0.60,
    "attempt1": 2.00,
    "verification": 1.80,
    "first_agent_message": 1.70
  },
  
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

#### Scripted Initialization Analysis
| Field | Description |
|-------|-------------|
| `conversation_type` | The provided conversation type string (e.g., `shared-phone`, `private-phone`, `web-audio`, `presentation`, or any custom string). Unknown/custom types are handled like web-audio but preserved as-is. |
| `initial_latency_points` | Dictionary of interval latencies in seconds for detected stages. Keys may include: `attempt2`, `attempt1`, `verification`, `disclaimer`, `first_agent_message`, `terminate`. |

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
## 🎬 Scripted Initialization Analysis Details

This analysis identifies early scripted transitions using template matching on the agent (right) channel and agent VAD timing. It produces interval latencies (in seconds) for key stages and adds them to `initial_latency_points`.

Supported conversation types (set per file input):
- `shared-phone`
- `private-phone`
- `web-audio`
- `presentation`

Type handling:
- For `shared-phone`, the analyzer tries to match templates `attempt2`, `attempt1`, `terminate` (and `redirect` ⇒ `disclaimer`) from `audio_files/` (MP3 preferred, WAV fallback). It merges agent segments with ≤1.5s gaps before matching. If no templates match, it falls back to: `verification` (first agent segment), `first_agent_message` (second agent segment).
- For `private-phone`, no template match is required. It reports two latencies: `verification` (to 1st agent segment), `first_agent_message` (from end of 1st to start of 2nd agent segment).
- For `web-audio` and `presentation`, it reports only `first_agent_message` latency (to the first agent segment).
- Auto-detection: If type is `web-audio`/`presentation` but templates appear in early segments, the analyzer switches to shared-phone logic automatically.

Matching method & logs:
- MFCC + DTW (cosine) distance; lower is better. Acceptance threshold is configurable.
- Logs print per-segment distances and an approximate similarity for each template, plus the best match for that segment.

Configuration knobs (in `scripted_analysis.py`):
- `TEMPLATE_EXTS`: order of template extensions to try (default `[".mp3", ".wav"]`).
- `DTW_ACCEPTANCE_THRESHOLD`: match acceptance threshold (default `0.0150`). Lower is stricter.
- `MAX_TEMPLATE_CHECKS`: number of leading agent segments to attempt matching (default `6`).
- `MIN_SEGMENT_SECONDS`: skip matching on very short segments (default `0.15`).
- `PRINT_SIMILARITY_SCORES`: print detailed per-template distances and best match (default `True`).
- Agent VAD segments are merged with a 1.5s max gap before matching.

Output examples:
- Shared-phone with attempt2 → attempt1 → verification → first_agent_message:
  `{ "attempt2": 0.60, "attempt1": 2.00, "verification": 1.80, "first_agent_message": 1.70 }`
- Shared-phone bad path (attempt2 → attempt1 → terminate):
  `{ "attempt2": 0.60, "attempt1": 2.00, "terminate": 1.50 }`
- Private-phone:
  `{ "verification": 0.80, "first_agent_message": 2.10 }`
- Web/presentation:
  `{ "first_agent_message": 0.75 }`

Providing type in requests:
- Include `conversation_type` per file:
  - `shared-phone`, `private-phone`, `web-audio`, `presentation`
  - Unknown values (e.g., `whatsapp`) are treated like `web-audio` but preserved in the response.

CSV batch usage:
- `test_input.csv` can include an optional `conversation_type` column. The batch logger forwards it to the API and includes `conversation_type` and `initial_latency_points` in the output CSV.

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
