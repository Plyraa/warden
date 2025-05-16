import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from audio_metrics import AudioMetricsCalculator
from visualization import AudioVisualizer
from database import init_db  # Ensure DB is initialized if web_ui is run directly

app = Flask(__name__)

# Initialize DB when app starts, if web_ui.py is the entry point
# This is a simple way; for more complex apps, consider Flask app context or CLI commands
init_db()


# Configure templates and static files
@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create template HTML file
with open("templates/index.html", "w") as f:
    f.write(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Center Audio Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 5px;
        }
        
        h1, h2, h3 {
            color: #333;
        }
        
        .file-selector {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        
        .visualization-section {
            margin-bottom: 30px;
        }
        
        .metrics-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .metrics-table th, .metrics-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #f2f2f2;
        }
        
        .visualization-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
        }
        
        select, button {
            padding: 8px 12px;
            margin-right: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        .loading {
            display: none;
            margin-top: 20px;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .audio-player {
            margin: 20px 0;
            display: none;
        }
        
        audio {
            width: 100%;
        }
        
        .transcript-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.5;
        }
          .overlap {
            background-color: #ffcccc;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .placeholder-notice {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background-color: #f1f1f1;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        
        .tab.active {
            background-color: #4CAF50;
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Call Center Audio Analysis</h1>
        
        <div class="file-selector">
            <h3>Select Audio File</h3>
            <form id="audioForm">
                <select id="audioFile" required>
                    <option value="">-- Select a file --</option>
                </select>
                <button type="submit">Analyze</button>
            </form>
        </div>
        
        <div class="audio-player" id="audioPlayer">
            <h3>Audio Playback</h3>
            <audio id="audioElement" controls></audio>
        </div>
        
        <div class="loading" id="loadingIndicator">
            <p>Processing audio... This may take a moment.</p>
            <div class="spinner"></div>
        </div>
        
        <div id="resultContainer" style="display: none;">
            <div class="tabs">
                <div class="tab active" data-tab="visualizations">Visualizations</div>
                <div class="tab" data-tab="transcript">Transcript</div>
            </div>
            
            <div class="tab-content active" id="visualizations">
                <div class="visualization-section">
                    <h2>Conversation Timeline</h2>
                    <p>This timeline shows when the user (left channel) and AI agent (right channel) were speaking during the call.</p>
                    <img id="timelineImage" class="visualization-image" src="" alt="Timeline visualization">
                </div>
                
                <div class="visualization-section">
                    <h2>Speech Overlap Analysis</h2>
                    <p>This visualization shows the word-level timing and highlights speech overlaps between the user and AI agent.</p>
                    <img id="speechOverlapImage" class="visualization-image" src="" alt="Speech overlap visualization">
                </div>
                
                <div class="visualization-section">
                    <h2>Waveform Visualization</h2>
                    <p>This shows the audio waveforms for both channels with speaking intervals highlighted.</p>
                    <img id="waveformImage" class="visualization-image" src="" alt="Waveform visualization">
                </div>
                
                <div class="visualization-section">
                    <h2>Response Latency Distribution</h2>
                    <p>This histogram shows the distribution of AI agent response latencies after user utterances.</p>
                    <img id="latencyHistImage" class="visualization-image" src="" alt="Latency histogram">
                </div>
                
                <div id="metricsTable">
                    <!-- Metrics table will be inserted here -->
                </div>
            </div>
            
            <div class="tab-content" id="transcript">
                <h2>Conversation Transcript</h2>
                <p>This is the transcribed conversation with speech overlaps highlighted in red.</p>
                <div id="transcriptContainer" class="transcript-container">
                    <!-- Transcript will be inserted here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            fetch('/get_audio_files')
                .then(response => response.json())
                .then(data => {
                    const selectElement = document.getElementById('audioFile');
                    data.forEach(file => {
                        const option = document.createElement('option');
                        option.value = file;
                        option.textContent = file;
                        selectElement.appendChild(option);
                    });
                });
                
            document.getElementById('audioForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const loadingIndicator = document.getElementById('loadingIndicator');
                loadingIndicator.style.display = 'block';
                
                const resultContainer = document.getElementById('resultContainer');
                resultContainer.style.display = 'none';
                
                const audioFile = document.getElementById('audioFile').value;
                
                fetch('/analyze_audio', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ audioFile: audioFile }),
                })
                .then(response => response.json())
                .then(data => {
                    // Update audio player
                    document.getElementById('audioElement').src = `/audio/${audioFile}`;
                    document.getElementById('audioPlayer').style.display = 'block';
                    
                    // Update images
                    document.getElementById('timelineImage').src = `data:image/png;base64,${data.timeline_img}`;
                    document.getElementById('waveformImage').src = `data:image/png;base64,${data.waveform_img}`;
                    document.getElementById('latencyHistImage').src = `data:image/png;base64,${data.latency_hist_img}`;
                    document.getElementById('speechOverlapImage').src = `data:image/png;base64,${data.speech_overlap_img}`;
                    
                    // Update metrics table
                    document.getElementById('metricsTable').innerHTML = data.metrics_table;
                      // Format and display transcript
                    const transcriptContainer = document.getElementById('transcriptContainer');
                    if (data.transcript_dialog) {
                        // Process transcript to highlight overlaps
                        const formattedTranscript = formatTranscriptWithOverlaps(data.transcript_dialog);
                        
                        // Add notice for placeholder transcripts
                        if (data.is_placeholder_transcript) {
                            transcriptContainer.innerHTML = 
                                '<div class="placeholder-notice">' +
                                '<strong>Note:</strong> This is a placeholder transcript generated from speech timing data. ' +
                                'No actual transcription was performed to save API credits. ' +
                                'The transcript shows the timing of speech segments rather than actual words.' +
                                '</div>' + formattedTranscript;
                        } else {
                            transcriptContainer.innerHTML = formattedTranscript;
                        }
                    } else {
                        transcriptContainer.innerHTML = '<em>No transcript available for this recording.</em>';
                    }
                    
                    // Show results
                    resultContainer.style.display = 'block';
                    loadingIndicator.style.display = 'none';
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingIndicator.style.display = 'none';
                    alert('An error occurred during analysis. Please try again.');
                });
            });
            
            // Tab functionality
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    // Remove active class from all tabs
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = this.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                });
            });
            
            // Function to format transcript with overlaps highlighted
            function formatTranscriptWithOverlaps(transcript) {
                // Look for words marked with [OVERLAP] tag and wrap them in span
                return transcript.replace(/(\\S+)\\[OVERLAP\\]/g, '<span class="overlap">$1</span>');
            }
        });
    </script>
</body>
</html>
    """
    )

# Create and initialize calculator and visualizer
calculator = AudioMetricsCalculator()
visualizer = AudioVisualizer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_audio_files")
def get_audio_files():
    audio_files = []
    for filename in os.listdir("stereo_test_calls"):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            audio_files.append(filename)
    return jsonify(audio_files)


@app.route("/audio/<filename>")
def serve_audio(filename):
    # Construct the expected downsampled filename
    base, ext = os.path.splitext(filename)
    downsampled_filename = base + "_downsampled" + ext  # Handles both .mp3 and .wav

    downsampled_path = os.path.join("sampled_test_calls", downsampled_filename)

    original_path_in_input_dir = os.path.join("stereo_test_calls", filename)

    if os.path.exists(downsampled_path):
        return send_from_directory("sampled_test_calls", downsampled_filename)
    elif os.path.exists(
        original_path_in_input_dir
    ):  # Serve original if not yet processed by web UI
        return send_from_directory("stereo_test_calls", filename)
    else:
        return "File not found", 404


@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    data = request.get_json()
    audio_file = data.get("audioFile")

    # Process the audio file
    metrics = calculator.process_file(audio_file)

    # Generate visualizations
    output_path = metrics["downsampled_path"]
    vis_data = visualizer.generate_web_visualization(metrics, output_path)

    # Add a flag for placeholder transcripts (generated from segments)
    if metrics.get("transcript_data") and metrics["transcript_data"].get(
        "generated_from_segments"
    ):
        vis_data["is_placeholder_transcript"] = True
    else:
        vis_data["is_placeholder_transcript"] = False

    return jsonify(vis_data)


if __name__ == "__main__":
    # init_db() # Already called above for when web_ui.py is run directly
    app.run(
        debug=False, port=5000
    )  # Changed debug to False as per previous linting suggestion
