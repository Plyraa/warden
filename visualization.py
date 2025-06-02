import base64
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np


class AudioVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        pass

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

            # Filter for only AI agent response latencies (user-to-agent transitions)
            ai_latencies = [
                item
                for item in latency_details
                if item.get("interaction_type", "").startswith("user_to_agent")
            ]

            if not ai_latencies:
                # No AI response latencies found
                ax.text(
                    0.5,
                    0.5,
                    "No AI agent response latencies found",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                plt.tight_layout()
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close(fig)
                return img_base64

            # Extract data - use to_turn_start for x-axis to show when AI responses occurred
            start_times = [item["to_turn_start"] for item in ai_latencies]
            latencies = [item["latency_seconds"] for item in ai_latencies]
            ratings = [item["rating"] for item in ai_latencies]

            # Number each AI response for clarity
            response_numbers = list(range(1, len(ai_latencies) + 1))

            # Define colors for different ratings
            rating_colors = {
                "Perfect": "#28a745",  # Green
                "Good": "#5cb85c",  # Light green
                "OK": "#ffc107",  # Yellow
                "Bad": "#f0ad4e",  # Orange
                "Poor": "#dc3545",  # Red
            }

            # Plot latency points
            scatter = ax.scatter(  # noqa: F841
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
                )  # Set labels and title
            title = "AI Agent Response Latency Timeline"
            if filename:
                title += f" - {filename}"
            ax.set_title(title)
            ax.set_xlabel("Conversation Time (seconds)")
            ax.set_ylabel("AI Response Latency (seconds)")

            # Set x-axis to show the full conversation duration
            ax.set_xlim(0, duration)

            # Set y-axis to show a bit more than the maximum latency or at least up to 6 seconds
            y_max = max(6, max(latencies) * 1.1) if latencies else 6
            ax.set_ylim(0, y_max)

            # Add grid
            ax.grid(True, alpha=0.3, zorder=1)

            # Add legend
            ax.legend(
                loc="upper right"
            )  # Add annotations for each point showing AI response number and latency
            for i, (x, y, rating, resp_num) in enumerate(
                zip(start_times, latencies, ratings, response_numbers)
            ):
                ax.annotate(
                    f"AI #{resp_num}: {y:.1f}s",
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

        # Get overlap data
        overlap_data = metrics.get("overlap_data", {})
        overlaps = overlap_data.get("overlaps", [])

        # Count overlaps by type
        ai_interrupting_count = 0
        user_interrupting_count = 0
        ai_interrupting_duration = 0
        user_interrupting_duration = 0

        # Process each overlap
        for overlap_item in overlaps:
            overlap_duration_val = float(overlap_item["duration"])
            interrupter = overlap_item["interrupter"]

            # Count interruptions by type
            if interrupter == "ai_agent":
                ai_interrupting_count += 1
                ai_interrupting_duration += overlap_duration_val
            elif interrupter == "user":
                user_interrupting_count += 1
                user_interrupting_duration += overlap_duration_val

        # Check if we have VAD metrics
        has_vad_metrics = vad_latency_metrics and all(
            value is not None for value in vad_latency_metrics.values()
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
                        <td>Average response latency for AI agent </td>
                    </tr>
                    <tr>
                        <td>Min Latency</td>
                        <td>{:.2f} s</td>
                        <td>Minimum response latency</td>
                    </tr>
                    <tr>
                        <td>Max Latency</td>
                        <td>{:.2f} s</td>
                        <td>Maximum response latency</td>
                    </tr>
                    <tr>
                        <td>P10 Latency</td>
                        <td>{:.2f} s</td>
                        <td>10th percentile response latency</td>
                    </tr>
                    <tr>
                        <td>P50 Latency (Median)</td>
                        <td>{:.2f} s</td>
                        <td>50th percentile response latency</td>
                    </tr>
                    <tr>
                        <td>P90 Latency</td>
                        <td>{:.2f} s</td>
                        <td>90th percentile response latency</td>
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
                    </tr>                    <tr>
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
                        <td>Agent Overlaps</td>
                        <td>{} (Duration: {:.2f}s)</td>
                        <td>Number of times the AI agent overlapped with user speech</td>
                    </tr>
                    <tr>
                        <td>User Overlaps</td>
                        <td>{} (Duration: {:.2f}s)</td>
                        <td>Number of times the user overlapped with AI agent speech</td>
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
            # Other metrics
            "Yes" if metrics["ai_interrupting_user"] else "No",
            "Yes" if metrics["user_interrupting_ai"] else "No",
            ai_interrupting_count,
            ai_interrupting_duration,
            user_interrupting_count,
            user_interrupting_duration,
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
        filename = metrics["filename"]

        transcript_data = metrics.get("transcript_data")
        transcript_dialog = "Transcript not available."
        if transcript_data and transcript_data.get("dialog"):
            transcript_dialog = transcript_data["dialog"]

        # Load audio to get duration
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)

        waveform_img = self.generate_waveform_plot(
            audio_path,
            metrics.get(
                "user_vad_segments", []
            ),  # Still need separated segments for waveform highlights
            metrics.get(
                "agent_vad_segments", []
            ),  # Still need separated segments for waveform highlights
        )

        metrics_table = self.generate_metrics_table_html(metrics)

        # Prepare data for speech overlap visualization
        transcript_with_overlap = transcript_data or {}

        # Ensure overlap_data is passed directly from metrics
        if "overlap_data" not in transcript_with_overlap and "overlap_data" in metrics:
            transcript_with_overlap = dict(
                transcript_with_overlap
            )  # Create a copy to avoid modifying the original
            transcript_with_overlap["overlap_data"] = metrics["overlap_data"]

        # Generate speech overlap visualization
        speech_overlap_img = self.generate_speech_overlap_visualization(
            transcript_with_overlap,
            metrics.get("user_vad_segments", []),
            metrics.get("agent_vad_segments", []),
            duration,
            filename,
        )

        # Generate VAD latency timeline (AI Start - User End)
        vad_latency_details = metrics.get("vad_latency_details", [])
        vad_latency_img = self.generate_vad_latency_timeline(
            vad_latency_details, duration, filename
        )
        
        return {
            "waveform_img": waveform_img,
            "speech_overlap_img": speech_overlap_img,
            "metrics_table": metrics_table,
            "vad_latency_img": vad_latency_img,
            "vad_latency_details": vad_latency_details,  # Add latency details for interactive chart
            "audio_duration": duration,  # Add duration for chart scaling
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
        if not transcript_data:
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

        # Get overlap data directly from metrics["overlap_data"]
        overlap_data = transcript_data.get("overlap_data", {})

        # Extract overlap information
        overlaps_list = overlap_data.get("overlaps", [])
        print("debug overlap print in visualization:", overlaps_list)
        total_overlap_count = overlap_data.get("total_overlap_count", 0)
        has_ai_interrupting_user = overlap_data.get("has_ai_interrupting_user", False)
        has_user_interrupting_ai = overlap_data.get("has_user_interrupting_ai", False)

        # If we have words, use them for word count statistics
        has_words = "words" in transcript_data
        if has_words:
            words = sorted(transcript_data.get("words", []), key=lambda w: w["start"])
            customer_words = [w for w in words if w["speaker"] == "customer"]
            agent_words = [w for w in words if w["speaker"] == "ai_agent"]
            customer_word_count = len(customer_words)
            agent_word_count = len(agent_words)
        else:
            words = []
            customer_word_count = 0
            agent_word_count = 0

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Set plot parameters
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["User (Customer)", "AI Agent"])
        ax.set_xlabel("Time (seconds)")

        # Create a more descriptive title
        title = "Speech Analysis"

        if filename:
            title += f" - {filename}"

        ax.set_title(title)

        # Add gridlines for better time correlation
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Add horizontal lines to separate speaker lanes
        ax.axhline(y=1.5, color="gray", linestyle="-", alpha=0.3, linewidth=1)

        # Add minor tick lines for better time precision
        minor_ticks = np.arange(0, duration, 1.0)  # 1 second intervals
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which="minor", axis="x", alpha=0.2)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="User Speech"),
            Patch(facecolor="green", alpha=0.7, label="AI Agent Speech"),
            Patch(facecolor="red", alpha=0.7, label="AI Interrupting User"),
            Patch(facecolor="yellow", alpha=0.7, label="User Interrupting AI"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Add tick marks for timing
        if has_words and words:
            # Use word timings if available
            ticks = []
            labels = []
            max_ticks = 20  # Limit number of ticks to avoid crowding
            tick_interval = max(1, len(words) // max_ticks)

            for i, word in enumerate(words):
                if i % tick_interval == 0:
                    ticks.append(word["start"])
                    labels.append(f"{word['start']:.1f}s")
        else:
            # Otherwise use regular intervals
            num_ticks = 10
            tick_interval = duration / num_ticks
            ticks = [i * tick_interval for i in range(num_ticks + 1)]
            labels = [f"{tick:.1f}s" for tick in ticks]

        # Add a secondary x-axis for timings
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.tick_params(axis="x", colors="gray")
        ax2.set_xlabel("Time (seconds)", color="gray")

        # Plot user speech segments
        for window in user_windows:
            start = float(window["start"])
            end = float(window["end"])
            ax.add_patch(
                plt.Rectangle(
                    (start, 1 - 0.1), end - start, 0.2, color="blue", alpha=0.7
                )
            )

        # Plot agent speech segments
        for window in agent_windows:
            start = float(window["start"])
            end = float(window["end"])
            ax.add_patch(
                plt.Rectangle(
                    (start, 2 - 0.1), end - start, 0.2, color="green", alpha=0.7
                )
            )

        # Calculate total durations for statistics
        total_user_duration = sum(
            float(window["end"]) - float(window["start"]) for window in user_windows
        )
        total_agent_duration = sum(
            float(window["end"]) - float(window["start"]) for window in agent_windows
        )
        total_audio_duration = duration  # In seconds

        # Plot overlaps with different colors based on who interrupted
        total_overlap_duration = 0
        ai_interrupting_count = 0
        user_interrupting_count = 0
        ai_interrupting_duration = 0
        user_interrupting_duration = 0

        for overlap_item in overlaps_list:
            start = float(overlap_item["start"])
            end = float(overlap_item["end"])
            overlap_duration_val = float(overlap_item["duration"])
            interrupter = overlap_item["interrupter"]
            interrupted_party = overlap_item["interrupted"]

            total_overlap_duration += overlap_duration_val

            # Count interruptions by type
            if interrupter == "ai_agent":
                ai_interrupting_count += 1
                ai_interrupting_duration += overlap_duration_val
            elif interrupter == "user":
                user_interrupting_count += 1
                user_interrupting_duration += overlap_duration_val

            color = None
            y_rect_base = None
            rect_height = 0.2  # Same height as speech segment bars
            alpha_val = 0.85  # Slightly more prominent for overlaps

            if interrupter == "ai_agent":  # AI interrupted User
                color = "red"
                if interrupted_party == "user":  # Draw on user's timeline
                    y_rect_base = 1 - (rect_height / 2)  # Centered on user's speech bar
                else:  # Default to middle if interrupted party not specified or different
                    y_rect_base = 1.5 - (rect_height / 2)
            elif interrupter == "user":  # User interrupted AI
                color = "yellow"
                if interrupted_party == "ai_agent":  # Draw on AI's timeline
                    y_rect_base = 2 - (rect_height / 2)  # Centered on AI's speech bar
                else:  # Default to middle
                    y_rect_base = 1.5 - (rect_height / 2)

            if color and y_rect_base is not None:
                ax.add_patch(
                    plt.Rectangle(
                        (start, y_rect_base),
                        overlap_duration_val,
                        rect_height,
                        color=color,
                        alpha=alpha_val,
                        zorder=4,  # Draw overlaps on top of speech, but below text/word markers
                    )
                )
                # Add text for the duration of the overlap, ensuring it's readable
                ax.text(
                    start + overlap_duration_val / 2,
                    y_rect_base
                    + rect_height / 2,  # Vertically centered on the overlap bar
                    f"{overlap_duration_val:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",  # Black text for better contrast on red/yellow
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", alpha=0.65
                    ),  # Background for text
                    zorder=5,  # Ensure text is above the overlap bar
                )

        # Add detailed statistics
        stats_text = "Overlapping Speech Analysis:\n"
        stats_text += f"• Total Overlaps: {total_overlap_count} instances ({total_overlap_duration:.2f}s)\n"

        if total_overlap_count > 0:
            overlap_percent = (total_overlap_duration / total_audio_duration) * 100
            stats_text += f"• Overlap: {overlap_percent:.1f}% of audio duration\n"

            if has_ai_interrupting_user:
                stats_text += f"• AI Interrupting User: {ai_interrupting_count} times ({ai_interrupting_duration:.2f}s)\n"

            if has_user_interrupting_ai:
                stats_text += f"• User Interrupting AI: {user_interrupting_count} times ({user_interrupting_duration:.2f}s)\n"

        # Add word count stats if available
        if has_words:
            stats_text += "\nWord Stats:\n"
            stats_text += f"• User Words: {customer_word_count}\n"
            stats_text += f"• AI Words: {agent_word_count}\n"

        # Add speaking time stats
        stats_text += "\nSpeaking Time:\n"
        stats_text += f"• User: {total_user_duration:.2f}s ({(total_user_duration / total_audio_duration) * 100:.1f}%)\n"
        stats_text += f"• AI: {total_agent_duration:.2f}s ({(total_agent_duration / total_audio_duration) * 100:.1f}%)"

        # Create a text box with statistics
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.01,
            0.99,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=props,
            linespacing=1.3,  # Increase line spacing for readability
        )

        # Convert to base64
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        return base64.b64encode(buf.read()).decode("utf-8")
