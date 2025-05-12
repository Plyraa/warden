import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from audio_metrics import AudioMetricsCalculator
from visualization import AudioVisualizer

app = Flask(__name__)


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
            text-align: center;
        }
        
        .audio-player {
            margin-top: 20px;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Call Center Audio Analysis</h1>
        
        <div class="file-selector">
            <h2>Select Audio File</h2>
            <form id="audioForm">
                <select id="audioFile" name="audioFile">
                    <!-- Options will be populated dynamically -->
                </select>
                <button type="submit">Analyze</button>
            </form>
            <div class="loading" id="loadingIndicator">
                <p>Processing audio... This may take a few moments.</p>
            </div>
        </div>
        
        <div id="audioPlayer" class="audio-player" style="display: none;">
            <h2>Audio Player</h2>
            <audio controls id="audioElement" style="width: 100%;">
                <source src="" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>
        
        <div id="resultContainer" style="display: none;">
            <div class="visualization-section">
                <h2>Conversation Timeline</h2>
                <p>This timeline shows when the user (left channel) and AI agent (right channel) were speaking during the call.</p>
                <img id="timelineImage" class="visualization-image" src="" alt="Timeline visualization">
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
                    
                    // Update metrics table
                    document.getElementById('metricsTable').innerHTML = data.metrics_table;
                    
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
    downsampled_filename = base + "_downsampled" + ext # Handles both .mp3 and .wav

    downsampled_path = os.path.join(
        "sampled_test_calls", downsampled_filename
    )
    
    original_path_in_input_dir = os.path.join("stereo_test_calls", filename)

    if os.path.exists(downsampled_path):
        return send_from_directory("sampled_test_calls", downsampled_filename)
    elif os.path.exists(original_path_in_input_dir): # Serve original if not yet processed by web UI
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

    return jsonify(vis_data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
