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

    # Add transcript words if available for chat bubble display
    if metrics.get("transcript_data") and "words" in metrics["transcript_data"]:
        vis_data["transcript_words"] = metrics["transcript_data"]["words"]
    else:
        vis_data["transcript_words"] = []

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
