# Audio Analysis for Jotform AI Agent Calls

This project provides tools for analyzing audio recordings of Jotform AI Agent calls. It can process audio files to extract various metrics and offers a web interface for visualizing these metrics.

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

### Web Interface (Recommended)

To start the web interface for visualizing audio metrics:

```bash
python .\warden.py --web
```

This will start the server at `http://127.0.0.1:5000`. You can then open this URL in your web browser to access the application.

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
