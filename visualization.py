import base64
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np


class AudioVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        pass

    def generate_channel_activity_plot(
        self, merged_turns, duration, filename=""
    ):
        """
        Generate a horizontal timeline showing user and agent activity based on merged_turns.

        Args:
            merged_turns: List of dictionaries, where each dictionary is a turn with
                          {'speaker': 'user'/'ai_agent', 'start': float, 'end': float}.
            duration: Total duration of the audio in seconds
            filename: Name of the file being visualized

        Returns:
            Base64 encoded image data
        """
        fig, ax = plt.subplots(figsize=(12, 3))

        # Set plot parameters
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["User", "AI Agent"]) # Simplified labels
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Conversation Timeline for {filename}")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Plot turns from merged_turns
        for turn in merged_turns:
            try:
                start = float(turn["start"])
                end = float(turn["end"])
                speaker = turn["speaker"]

                if speaker == "user":
                    ax.add_patch(
                        plt.Rectangle((start, 0.9), end - start, 0.2, color="blue", alpha=0.7)
                    )
                elif speaker == "ai_agent":
                    ax.add_patch(
                        plt.Rectangle((start, 1.9), end - start, 0.2, color="green", alpha=0.7)
                    )
            except (KeyError, TypeError, ValueError) as e:
                print(
                    f"ERROR: Could not process turn in generate_channel_activity_plot: {turn}. Error: {e}"
                )
                continue

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)

        # Encode the image as base64
        img_data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return img_data

    def generate_waveform_plot(self, audio_path, user_windows, agent_windows):
        """
        Generate a waveform plot with user and agent activity highlighted

        Args:
            audio_path: Path to audio file
            user_windows: List of (start_time, end_time) tuples for user channel
            agent_windows: List of (start_time, end_time) tuples for agent channel

        Returns:
            Base64 encoded image data
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)

        # Create plot with two subplots (one for each channel)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Time values for x-axis
        times = np.linspace(0, duration, y.shape[1])

        # Plot left channel (user)
        axes[0].plot(times, y[0], color="blue", alpha=0.6)
        axes[0].set_title("Left Channel (User)")

        # Highlight user speaking intervals
        for start, end in user_windows:
            axes[0].axvspan(start, end, color="blue", alpha=0.2)

        # Plot right channel (agent)
        axes[1].plot(times, y[1], color="green", alpha=0.6)
        axes[1].set_title("Right Channel (AI Agent)")

        # Highlight agent speaking intervals
        for start, end in agent_windows:
            axes[1].axvspan(start, end, color="green", alpha=0.2)

        # Set common x-axis label
        axes[1].set_xlabel("Time (seconds)")

        # Set common y-axis label
        fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)

        # Encode the image as base64
        img_data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return img_data

    def generate_latency_histogram(self, latencies):
        """
        Generate a histogram of agent response latencies

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Base64 encoded image data
        """
        if not latencies:
            # Create an empty plot if no latencies
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No latency data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create histogram
            ax.hist(latencies, bins=10, color="skyblue", edgecolor="black")

            ax.set_title("Distribution of AI Agent Response Latencies")
            ax.set_xlabel("Latency (milliseconds)")
            ax.set_ylabel("Frequency")

            # Add average line
            avg_latency = sum(latencies) / len(latencies)
            ax.axvline(
                avg_latency,
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Average: {avg_latency:.2f} ms",
            )
            ax.legend()

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)

        # Encode the image as base64
        img_data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return img_data

    def generate_vad_latency_timeline(self, latency_details, duration, filename=""):
        """
        Generate a visualization of all latency measurements throughout the conversation

        Args:
            latency_details: List of latency detail dictionaries
            duration: Total duration of the audio in seconds
            filename: Optional filename for the title

        Returns:
            Base64 encoded image data
        """
        if not latency_details:
            # Create an empty plot if no data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(
                0.5,
                0.5,
                "No VAD latency data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract data - use agent_start for x-axis to show when responses occurred
            start_times = [item["agent_start"] for item in latency_details]
            latencies = [item["latency_seconds"] for item in latency_details]
            ratings = [item["rating"] for item in latency_details]
            
            # Number each response for clarity
            response_numbers = list(range(1, len(latency_details) + 1))

            # Define colors for different ratings
            rating_colors = {
                "Perfect": "#28a745",  # Green
                "Good": "#5cb85c",  # Light green
                "OK": "#ffc107",  # Yellow
                "Bad": "#f0ad4e",  # Orange
                "Poor": "#dc3545",  # Red
            }

            # Plot latency points
            scatter = ax.scatter(
                start_times,
                latencies,
                c=[rating_colors.get(rating, "#007bff") for rating in ratings],
                s=100,  # Point size
                alpha=0.8,
                edgecolors="black",
                zorder=3,
            )

            # Add horizontal lines for thresholds
            thresholds = [(2, "Perfect"), (3, "Good"), (4, "OK"), (5, "Bad")]
            for threshold, label in thresholds:
                ax.axhline(
                    y=threshold,
                    linestyle="--",
                    alpha=0.7,
                    color=rating_colors.get(label, "gray"),
                    label=f"{label} threshold ({threshold}s)",
                    zorder=2,
                )

            # Set labels and title
            title = f"VAD-based Turn-taking Latency Timeline"
            if filename:
                title += f" - {filename}"
            ax.set_title(title)
            ax.set_xlabel("Conversation Time (seconds)")
            ax.set_ylabel("Response Latency (seconds)")

            # Set x-axis to show the full conversation duration
            ax.set_xlim(0, duration)

            # Set y-axis to show a bit more than the maximum latency or at least up to 6 seconds
            y_max = max(6, max(latencies) * 1.1) if latencies else 6
            ax.set_ylim(0, y_max)

            # Add grid
            ax.grid(True, alpha=0.3, zorder=1)

            # Add legend
            ax.legend(loc="upper right")

            # Add annotations for each point showing response number and latency
            for i, (x, y, rating, resp_num) in enumerate(zip(start_times, latencies, ratings, response_numbers)):
                ax.annotate(
                    f"#{resp_num}: {y:.1f}s",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)

        # Encode the image as base64
        img_data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return img_data

    def generate_metrics_table_html(self, metrics):
        """
        Generate HTML for metrics table

        Args:
            metrics: Dictionary containing calculated metrics

        Returns:
            HTML string for metrics table
        """
        vad_latency_metrics = metrics.get("vad_latency_metrics", {})

        # Check if we have VAD metrics
        has_vad_metrics = (
            vad_latency_metrics and all(value is not None for value in vad_latency_metrics.values())
        )

        # Function to get latency rating based on seconds
        def get_latency_rating(seconds):
            if seconds < 2:
                return "Perfect"
            elif seconds < 3:
                return "Good"
            elif seconds < 4:
                return "OK"
            elif seconds < 5:
                return "Bad"
            else:
                return "Poor"

        # Function to get color class based on rating
        def get_rating_color_class(rating):
            if rating == "Perfect":
                return "rating-perfect"
            elif rating == "Good":
                return "rating-good"
            elif rating == "OK":
                return "rating-ok"
            elif rating == "Bad":
                return "rating-bad"
            else:
                return "rating-poor"

        # Rating for average latency
        avg_rating = get_latency_rating(
            vad_latency_metrics.get("avg_latency", 0) if has_vad_metrics else 0
        )
        avg_color_class = get_rating_color_class(avg_rating)

        html = """
        <div class="metrics-container">
            <h3>Audio Analysis Metrics</h3>
            <style>
                .rating-perfect {{ color: #28a745; font-weight: bold; }}
                .rating-good {{ color: #5cb85c; font-weight: bold; }}
                .rating-ok {{ color: #ffc107; font-weight: bold; }}
                .rating-bad {{ color: #f0ad4e; font-weight: bold; }}
                .rating-poor {{ color: #dc3545; font-weight: bold; }}
                .metrics-section {{ margin-top: 20px; }}
            </style>

            <div class="metrics-section">
                <h4>VAD-based Turn-Taking Latency (AI Start - User End)</h4>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>Average Latency</td>
                        <td>{:.2f} s <span class="{}">({} rating)</span></td>
                        <td>Average response latency for AI agent (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>Min Latency</td>
                        <td>{:.2f} s</td>
                        <td>Minimum response latency (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>Max Latency</td>
                        <td>{:.2f} s</td>
                        <td>Maximum response latency (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>P10 Latency</td>
                        <td>{:.2f} s</td>
                        <td>10th percentile response latency (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>P50 Latency (Median)</td>
                        <td>{:.2f} s</td>
                        <td>50th percentile response latency (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>P90 Latency</td>
                        <td>{:.2f} s</td>
                        <td>90th percentile response latency (interrupted turns excluded)</td>
                    </tr>
                    <tr>
                        <td>AI Interruptions Handled</td>
                        <td>{}</td>
                        <td>Number of AI interruptions skipped for latency calculation</td>
                    </tr>
                </table>
            </div>

            <div class="metrics-section">
                <h4>Other Metrics</h4>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>AI Interrupting User</td>
                        <td>{}</td>
                        <td>Whether AI interrupted the user</td>
                    </tr>
                    <tr>
                        <td>User Interrupting AI</td>
                        <td>{}</td>
                        <td>Whether user interrupted the AI</td>
                    </tr>
                    <tr>
                        <td>Talk Ratio (Agent/User)</td>
                        <td>{:.2f}</td>
                        <td>Ratio of AI agent speaking time to user speaking time</td>
                    </tr>
                    <tr>
                        <td>Average Pitch</td>
                        <td>{:.2f} Hz</td>
                        <td>Average pitch of AI agent's voice</td>
                    </tr>
                    <tr>
                        <td>Words Per Minute</td>
                        <td>{:.2f} WPM</td>
                        <td>Estimated speech speed of AI agent</td>
                    </tr>
                </table>            </div>
        </div>
        """.format(
            # VAD metrics
            vad_latency_metrics.get("avg_latency", 0) if has_vad_metrics else 0,
            avg_color_class,
            avg_rating,
            vad_latency_metrics.get("min_latency", 0) if has_vad_metrics else 0,
            vad_latency_metrics.get("max_latency", 0) if has_vad_metrics else 0,
            vad_latency_metrics.get("p10_latency", 0) if has_vad_metrics else 0,
            vad_latency_metrics.get("p50_latency", 0) if has_vad_metrics else 0,
            vad_latency_metrics.get("p90_latency", 0) if has_vad_metrics else 0,
            vad_latency_metrics.get("ai_interruptions_handled_in_latency", 0) if has_vad_metrics else 0, # New metric
            # Other metrics
            "Yes" if metrics["ai_interrupting_user"] else "No",
            "Yes" if metrics["user_interrupting_ai"] else "No",
            metrics["talk_ratio"],
            metrics["average_pitch"],
            metrics["words_per_minute"],
        )

        return html

    def generate_web_visualization(self, metrics, audio_path):
        """
        Generate a dictionary with all visualization components for web UI

        Args:
            metrics: Dictionary containing calculated metrics
            audio_path: Path to the audio file

        Returns:
            Dictionary with visualization components
        """
        # Extract necessary data
        merged_turns = metrics.get("combined_speaker_turns", []) # Use merged_turns for the timeline
        filename = metrics["filename"]

        transcript_data = metrics.get("transcript_data")
        transcript_dialog = "Transcript not available."
        if transcript_data and transcript_data.get("dialog"):
            transcript_dialog = transcript_data["dialog"]

        # Load audio to get duration
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)

        # Generate visualizations
        timeline_img = self.generate_channel_activity_plot(
            merged_turns, duration, filename # Use merged_turns now
        )

        waveform_img = self.generate_waveform_plot(
            audio_path, 
            metrics.get("user_vad_segments", []), # Still need separated segments for waveform highlights
            metrics.get("agent_vad_segments", []) # Still need separated segments for waveform highlights
        )

        metrics_table = self.generate_metrics_table_html(metrics)

        # Generate speech overlap visualization if transcript data is available
        speech_overlap_img = self.generate_speech_overlap_visualization(
            transcript_data, 
            metrics.get("user_vad_segments", []), 
            metrics.get("agent_vad_segments", []), 
            duration, 
            filename
        )
        
        # Generate VAD latency timeline (AI Start - User End)
        vad_latency_details = metrics.get("vad_latency_details", [])
        vad_latency_img = self.generate_vad_latency_timeline(vad_latency_details, duration, filename)

        return {
            "timelineImage": timeline_img,
            "waveform_img": waveform_img,
            "speech_overlap_img": speech_overlap_img,
            "metrics_table": metrics_table,
            "vad_latency_img": vad_latency_img,
            "filename": filename,
            "transcript_dialog": transcript_dialog,
        }

    def generate_speech_overlap_visualization(
        self, transcript_data, user_windows, agent_windows, duration, filename=""
    ):
        """Generate a visualization showing speech overlaps between user and agent.

        Args:
            transcript_data: Transcript data dictionary with words and overlap info
            user_windows: List of (start_time, end_time) tuples for user channel
            agent_windows: List of (start_time, end_time) tuples for agent channel
            duration: Total duration of the audio in seconds
            filename: Name of the file being visualized

        Returns:
            Base64 encoded image data
        """
        if not transcript_data or "words" not in transcript_data:
            # Return empty visualization if no transcript data
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(
                0.5,
                0.5,
                "No transcript data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Convert to base64
            buf = BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            return base64.b64encode(buf.read()).decode("utf-8")

        # Sort words by start time
        words = sorted(transcript_data.get("words", []), key=lambda w: w["start"])

        # Calculate speakers' word counts and overlap stats
        customer_words = [w for w in words if w["speaker"] == "customer"]
        agent_words = [w for w in words if w["speaker"] == "ai_agent"]
        overlap_words = [w for w in words if w.get("is_overlap", False)]

        customer_word_count = len(customer_words)
        agent_word_count = len(agent_words)
        overlap_count = len(overlap_words)
        overlap_percentage = 0
        if customer_word_count + agent_word_count > 0:
            overlap_percentage = (
                overlap_count / (customer_word_count + agent_word_count)
            ) * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Set plot parameters
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["User (Customer)", "AI Agent"])
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Speech Analysis with Overlaps for {filename}")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="User Speech"),
            Patch(facecolor="green", alpha=0.7, label="AI Agent Speech"),
            Patch(facecolor="red", alpha=0.7, label="Speech Overlap"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Track where we've placed text to avoid overlapping labels
        text_positions = {
            "customer": [],  # List of (start, end) x-positions for text
            "ai_agent": [],
        }

        # Add tick marks for word timing
        word_ticks = []
        word_labels = []
        max_ticks = 20  # Limit number of ticks to avoid crowding
        tick_interval = max(1, len(words) // max_ticks)

        for i, word in enumerate(words):
            if i % tick_interval == 0:
                word_ticks.append(word["start"])
                word_labels.append(f"{word['start']:.1f}s")

        # Add a secondary x-axis for word timings
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(word_ticks)
        ax2.set_xticklabels(word_labels, rotation=45)
        ax2.tick_params(axis="x", colors="gray")
        ax2.set_xlabel("Word Start Times (seconds)", color="gray")

        # Plot words
        for word in words:
            start = word["start"]
            end = word["end"]
            speaker = word["speaker"]
            is_overlap = word.get("is_overlap", False)

            # Determine y-position based on speaker
            if speaker == "customer":
                y_pos = 1
            else:  # ai_agent
                y_pos = 2

            # Determine color based on overlap
            color = (
                "red" if is_overlap else ("blue" if speaker == "customer" else "green")
            )

            # Draw rectangle for the word
            ax.add_patch(
                plt.Rectangle(
                    (start, y_pos - 0.1), end - start, 0.2, color=color, alpha=0.7
                )
            )

        # Add statistics
        overlap_count = transcript_data.get("overlap_count", 0)
        has_overlaps = transcript_data.get("has_overlaps", False)

        stats_text = f"Overlapping Speech: {'Yes' if has_overlaps else 'No'}\n"
        stats_text += f"Overlap Count: {overlap_count} words\n"
        stats_text += (
            f"User Words: {customer_word_count}, Agent Words: {agent_word_count}\n"
        )
        stats_text += f"Overlap Percentage: {overlap_percentage:.1f}% of words"

        ax.text(
            0.01,
            0.01,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Convert to base64
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        return base64.b64encode(buf.read()).decode("utf-8")
