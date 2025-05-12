import os
import statistics

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class AudioMetricsCalculator:
    def __init__(self, input_dir="stereo_test_calls", output_dir="sampled_test_calls"):
        """Initialize the calculator with input and output directories"""
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def downsample_audio(self, input_file, target_sr=16000):
        """Downsample stereo audio file to target sample rate and return left and right channels"""
        # Get full path
        input_path = os.path.join(self.input_dir, input_file)
        output_path = os.path.join(self.output_dir, input_file.replace(".mp3", ".wav"))

        # Load audio
        audio, sr = librosa.load(input_path, sr=None, mono=False)

        # If audio is mono, duplicate to create stereo
        if len(audio.shape) == 1:
            audio = np.array([audio, audio])

        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Save downsampled audio
        sf.write(output_path, audio.T, target_sr)

        return audio, target_sr, output_path

    def calculate_activity_windows(
        self, audio, sr, threshold=-35, min_silence_duration=0.2
    ):
        """
        Calculate activity windows for each channel
        Returns lists of (start_time, end_time) tuples for each channel
        """
        # Convert threshold from dB to amplitude
        amplitude_threshold = 10 ** (threshold / 20)

        # Calculate window parameters
        min_silence_samples = int(min_silence_duration * sr)

        # Process each channel
        activity_windows = []

        for channel_idx in range(2):  # Assuming stereo audio
            channel_data = audio[channel_idx]

            # Get amplitude envelope using RMS with small window
            hop_length = int(0.01 * sr)  # 10 ms window hop
            window_length = int(0.025 * sr)  # 25 ms window length

            # Calculate RMS energy in small windows
            energy = librosa.feature.rms(
                y=channel_data, frame_length=window_length, hop_length=hop_length
            )[0]

            # Time values for each energy value
            times = librosa.times_like(energy, sr=sr, hop_length=hop_length)

            # Find active segments
            is_active = energy > amplitude_threshold

            # Convert to time windows
            channel_windows = []
            current_window = None

            for i, active in enumerate(is_active):
                if active and current_window is None:
                    # Start new window
                    current_window = [times[i], None]

                elif not active and current_window is not None:
                    # End current window if silence is long enough
                    if i > 0 and sum(is_active[i - min_silence_samples : i]) == 0:
                        current_window[1] = times[i - 1]
                        channel_windows.append(tuple(current_window))
                        current_window = None

            # Handle case where audio ends while active
            if current_window is not None:
                current_window[1] = times[-1]
                channel_windows.append(tuple(current_window))

            activity_windows.append(channel_windows)

        return activity_windows

    def calculate_average_latency(self, user_windows, agent_windows):
        """Calculate the average, p10, p50, and p90 latency between user utterances and agent responses"""
        latencies = self.calculate_agent_answer_latency(user_windows, agent_windows)

        if not latencies:
            return {
                "avg_latency": 0,
                "p10_latency": 0,
                "p50_latency": 0,
                "p90_latency": 0,
            }

        avg_latency = sum(latencies) / len(latencies) * 1000  # convert to ms

        # Calculate percentiles
        p10_latency = np.percentile(latencies, 10) * 1000
        p50_latency = np.percentile(latencies, 50) * 1000
        p90_latency = np.percentile(latencies, 90) * 1000

        return {
            "avg_latency": avg_latency,
            "p10_latency": p10_latency,
            "p50_latency": p50_latency,
            "p90_latency": p90_latency,
        }

    def calculate_agent_answer_latency(self, user_windows, agent_windows):
        """Calculate latency for each agent response after user utterances"""
        latencies = []

        for user_window in user_windows:
            user_end = user_window[1]

            # Find the next agent window that starts after this user window ends
            next_agent_windows = [w for w in agent_windows if w[0] > user_end]

            if next_agent_windows:
                # Get the closest agent window
                next_agent = min(next_agent_windows, key=lambda w: w[0])
                latency = next_agent[0] - user_end
                latencies.append(latency)

        return latencies

    def detect_ai_interrupting_user(self, user_windows, agent_windows):
        """Check if AI agent interrupts user during conversation"""
        for user_window in user_windows:
            user_start, user_end = user_window

            # Check if any agent window overlaps with user window
            for agent_start, agent_end in agent_windows:
                # Agent started speaking while user was speaking
                if agent_start > user_start and agent_start < user_end:
                    return True

        return False

    def detect_user_interrupting_ai(self, user_windows, agent_windows):
        """Check if user interrupts AI agent during conversation"""
        for agent_window in agent_windows:
            agent_start, agent_end = agent_window

            # Check if any user window overlaps with agent window
            for user_start, user_end in user_windows:
                # User started speaking while agent was speaking
                if user_start > agent_start and user_start < agent_end:
                    return True

        return False

    def calculate_talk_ratio(self, user_windows, agent_windows):
        """Calculate ratio of agent speaking time to user speaking time"""
        user_duration = sum(end - start for start, end in user_windows)
        agent_duration = sum(end - start for start, end in agent_windows)

        if user_duration == 0:
            return float("inf")  # Avoid division by zero

        return agent_duration / user_duration

    def calculate_average_pitch(self, audio, sr):
        """Calculate average pitch for agent channel (right channel)"""
        agent_audio = audio[1]

        # Calculate pitch using librosa's pitch tracking
        pitches, magnitudes = librosa.piptrack(y=agent_audio, sr=sr)

        # Only consider pitches with high magnitude
        pitches_flat = pitches[magnitudes > np.median(magnitudes)]

        # Filter out extreme values (noise)
        valid_pitches = pitches_flat[(pitches_flat > 50) & (pitches_flat < 500)]

        if len(valid_pitches) > 0:
            return np.mean(valid_pitches)
        else:
            return 0

    def calculate_words_per_minute(self, audio, sr, agent_windows):
        """
        Estimate words per minute for agent speech

        This is an approximation based on energy fluctuations in speech,
        since actual speech-to-text would be more complex.
        """
        agent_audio = audio[1]

        # Extract audio segments where agent is speaking
        agent_speech = []
        for start, end in agent_windows:
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            agent_speech.extend(agent_audio[start_idx:end_idx])

        if not agent_speech:
            return 0

        agent_speech = np.array(agent_speech)

        # Count syllables using energy peaks
        hop_length = int(0.01 * sr)  # 10ms hop
        frame_length = int(0.025 * sr)  # 25ms window

        # Calculate RMS energy
        energy = librosa.feature.rms(
            y=agent_speech, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Find peaks in energy (syllables)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(energy, distance=5, prominence=0.01)

        # Estimate syllables
        syllable_count = len(peaks)

        # Estimate words (roughly 1.5 syllables per word in English)
        word_count = syllable_count / 1.5

        # Calculate total agent speaking time in minutes
        total_speaking_time = sum(end - start for start, end in agent_windows) / 60

        if total_speaking_time > 0:
            return word_count / total_speaking_time
        else:
            return 0

    def process_file(self, filename):
        """Process a single audio file and calculate all metrics"""
        # Downsample audio file
        audio, sr, output_path = self.downsample_audio(filename)

        # Get activity windows for both channels
        activity_windows = self.calculate_activity_windows(audio, sr)

        # Right channel (index 1) is AI agent, Left channel (index 0) is user
        user_windows = activity_windows[0]  # Left channel
        agent_windows = activity_windows[1]  # Right channel

        # Calculate metrics
        metrics = {
            "filename": filename,
            "downsampled_path": output_path,
            "latency_metrics": self.calculate_average_latency(
                user_windows, agent_windows
            ),
            "agent_answer_latencies": [
                lat * 1000
                for lat in self.calculate_agent_answer_latency(
                    user_windows, agent_windows
                )
            ],  # ms
            "ai_interrupting_user": self.detect_ai_interrupting_user(
                user_windows, agent_windows
            ),
            "user_interrupting_ai": self.detect_user_interrupting_ai(
                user_windows, agent_windows
            ),
            "talk_ratio": self.calculate_talk_ratio(user_windows, agent_windows),
            "average_pitch": self.calculate_average_pitch(audio, sr),
            "words_per_minute": self.calculate_words_per_minute(
                audio, sr, agent_windows
            ),
            "user_windows": user_windows,
            "agent_windows": agent_windows,
        }

        return metrics
