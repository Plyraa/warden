import base64
import os
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np


class AudioVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        pass

    def generate_channel_activity_plot(
        self, user_windows, agent_windows, duration, filename=""
    ):
        """
        Generate a horizontal timeline showing user and agent activity

        Args:
            user_windows: List of (start_time, end_time) tuples for user channel
            agent_windows: List of (start_time, end_time) tuples for agent channel
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
        ax.set_yticklabels(["User (L)", "AI Agent (R)"])
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Channel Activity for {filename}")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Plot user activity (bottom row)
        for start, end in user_windows:
            ax.add_patch(
                plt.Rectangle((start, 0.9), end - start, 0.2, color="blue", alpha=0.7)
            )

        # Plot agent activity (top row)
        for start, end in agent_windows:
            ax.add_patch(
                plt.Rectangle((start, 1.9), end - start, 0.2, color="green", alpha=0.7)
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

    def generate_metrics_table_html(self, metrics):
        """
        Generate HTML for metrics table

        Args:
            metrics: Dictionary containing calculated metrics

        Returns:
            HTML string for metrics table
        """
        latency_metrics = metrics["latency_metrics"]

        html = """
        <div class="metrics-container">
            <h3>Audio Analysis Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Average Latency</td>
                    <td>{:.2f} ms</td>
                    <td>Average response latency for AI agent</td>
                </tr>
                <tr>
                    <td>P10 Latency</td>
                    <td>{:.2f} ms</td>
                    <td>10th percentile response latency</td>
                </tr>
                <tr>
                    <td>P50 Latency (Median)</td>
                    <td>{:.2f} ms</td>
                    <td>50th percentile response latency</td>
                </tr>
                <tr>
                    <td>P90 Latency</td>
                    <td>{:.2f} ms</td>
                    <td>90th percentile response latency</td>
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
            </table>
        </div>
        """.format(
            latency_metrics["avg_latency"],
            latency_metrics["p10_latency"],
            latency_metrics["p50_latency"],
            latency_metrics["p90_latency"],
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
        user_windows = metrics["user_windows"]
        agent_windows = metrics["agent_windows"]
        filename = metrics["filename"]
        latencies = metrics["agent_answer_latencies"]

        # Load audio to get duration
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)

        # Generate visualizations
        timeline_img = self.generate_channel_activity_plot(
            user_windows, agent_windows, duration, filename
        )

        waveform_img = self.generate_waveform_plot(
            audio_path, user_windows, agent_windows
        )

        latency_hist_img = self.generate_latency_histogram(latencies)

        metrics_table = self.generate_metrics_table_html(metrics)

        return {
            "timeline_img": timeline_img,
            "waveform_img": waveform_img,
            "latency_hist_img": latency_hist_img,
            "metrics_table": metrics_table,
            "filename": filename,
        }
