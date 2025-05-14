import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment  # Added for MP3 handling
import tempfile
from pathlib import Path
import concurrent.futures
from dotenv import load_dotenv  # Added for .env file support
import subprocess  # Added for direct ffmpeg calls

# ElevenLabs
from elevenlabs.client import ElevenLabs

# Database imports
from database import (
    get_db,
    add_analysis,
    get_analysis_by_filename,
    recreate_metrics_from_db,
)

# Load environment variables from .env file
load_dotenv()


class AudioMetricsCalculator:
    def __init__(self, input_dir="stereo_test_calls", output_dir="sampled_test_calls"):
        """Initialize the calculator with input and output directories"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.elevenlabs_client = None
        # ELEVENLABS_API_KEY is now loaded from .env by load_dotenv()
        # os.getenv will retrieve it if it was successfully loaded into the environment
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_api_key:
            self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        else:
            print(
                "WARNING: ELEVENLABS_API_KEY not found in .env file or environment. Transcription will be skipped."
            )

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def downsample_audio(self, input_file, target_sr=16000):
        """Downsample stereo audio file to target sample rate and return left and right channels, preserving format."""
        input_path = os.path.join(self.input_dir, input_file)
        output_filename = (
            os.path.splitext(input_file)[0]
            + "_downsampled"
            + os.path.splitext(input_file)[1]
        )
        output_path = os.path.join(self.output_dir, output_filename)

        # Check if downsampled file already exists
        if os.path.exists(output_path):
            print(f"Found existing downsampled file: {output_path}")
            # Ensure consistent return type with processing case
            audio, sr = librosa.load(output_path, sr=target_sr, mono=False)
            # Ensure audio is in the expected shape (channels, samples)
            if len(audio.shape) == 1:
                audio = np.array([audio, audio])
            elif audio.shape[0] != 2 and audio.shape[1] == 2:  # if (samples, channels)
                audio = audio.T
            return audio, sr, output_path

        print(f"Downsampling {input_file} to {output_path}")
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension == ".mp3":
            audio_segment = AudioSegment.from_mp3(input_path)
            # Ensure stereo
            if audio_segment.channels == 1:
                audio_segment = audio_segment.set_channels(2)

            # Resample if necessary
            if audio_segment.frame_rate != target_sr:
                audio_segment = audio_segment.set_frame_rate(target_sr)

            audio_segment.export(output_path, format="mp3")

            # For consistency with librosa's output, load the downsampled mp3 with librosa
            # This is for the return value, the file is already saved.
            audio, sr = librosa.load(output_path, sr=target_sr, mono=False)
            # Ensure audio is in the expected shape (channels, samples)
            if (
                len(audio.shape) == 1
            ):  # If mono after librosa load (should not happen with pydub export)
                audio = np.array([audio, audio])
            elif audio.shape[0] != 2 and audio.shape[1] == 2:  # if (samples, channels)
                audio = audio.T

        elif file_extension == ".wav":
            # Load audio
            audio, sr = librosa.load(input_path, sr=None, mono=False)

            # If audio is mono, duplicate to create stereo
            if len(audio.shape) == 1:
                audio = np.array([audio, audio])
            # Ensure audio is in the shape (channels, samples) before resample
            elif audio.shape[0] != 2 and audio.shape[1] == 2:  # if (samples, channels)
                audio = audio.T

            # Resample if necessary
            if sr != target_sr:
                # librosa.resample expects (channels, samples) or (samples,)
                # if audio.shape[0] != 2: # If it's (samples, 2)
                #     audio = audio.T # Transpose to (2, samples)
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Save downsampled audio
            # soundfile.write expects (samples, channels)
            sf.write(output_path, audio.T, target_sr)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return audio, target_sr, output_path
        
    def _split_stereo_audio(self, stereo_audio_path: str, temp_dir: str):
        """Splits a stereo audio file into two mono files (customer and agent) using subprocess.run with pan filter."""
        original_path = Path(stereo_audio_path)
        base_name = original_path.stem
        original_extension = original_path.suffix  # e.g., .mp3 or .wav

        # Output files will have the same extension as the input
        customer_path = Path(temp_dir) / f"{base_name}_customer{original_extension}"
        agent_path = Path(temp_dir) / f"{base_name}_agent{original_extension}"

        # Use pan filter to extract the left channel (customer)
        customer_command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", str(stereo_audio_path),
            "-filter_complex", "pan=mono|c0=c0",  # Extract left channel
            "-ac", "1",  # Ensure mono output
            str(customer_path)
        ]
        
        # Use pan filter to extract the right channel (agent)
        agent_command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", str(stereo_audio_path),
            "-filter_complex", "pan=mono|c0=c1",  # Extract right channel
            "-ac", "1",  # Ensure mono output
            str(agent_path)
        ]

        try:
            # Process left channel (customer)
            # print(f"Executing ffmpeg command: {' '.join(customer_command)}") # For debugging
            subprocess.run(customer_command, capture_output=True, text=True, check=True)
            
            # Process right channel (agent)
            # print(f"Executing ffmpeg command: {' '.join(agent_command)}") # For debugging
            subprocess.run(agent_command, capture_output=True, text=True, check=True)
            
            return str(customer_path), str(agent_path)
        except subprocess.CalledProcessError as e:
            print(
                f"Error splitting audio channels for {stereo_audio_path} using subprocess:"
            )
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            return None, None
        except FileNotFoundError:
            print(
                "Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH."
            )
            return None, None

    def _transcribe_channel(self, audio_path: str, speaker_label: str):
        """Transcribes a single mono audio file using ElevenLabs Scribe."""
        if not self.elevenlabs_client:
            return []  # Return empty list if client not initialized

        try:
            with open(audio_path, "rb") as f:
                response = self.elevenlabs_client.speech_to_text.convert(
                    file=f,
                    model_id="scribe_v1",  # Or your preferred model
                    diarize=False,
                    num_speakers=1,
                    timestamps_granularity="word",
                    tag_audio_events=False,  # Set to True if you need event tags
                )

            words = (
                response.words if hasattr(response, "words") and response.words else []
            )
            # Attach our own speaker label
            for word_info in words:
                word_info["speaker"] = speaker_label
            return words
        except Exception as e:
            print(f"Error during transcription for {audio_path} ({speaker_label}): {e}")
            return []

    def _merge_and_format_transcript(self, words_customer, words_agent):
        """Merges word lists from customer and agent, sorts them, and formats into a dialog."""
        all_words = sorted(words_customer + words_agent, key=lambda w: w["start"])

        dialog_parts = []
        current_buffer, current_speaker = [], None
        for word_info in all_words:
            word_text = word_info["text"]
            speaker = word_info["speaker"]
            if speaker != current_speaker:
                if current_buffer:
                    dialog_parts.append(
                        f"{current_speaker.upper()}: {' '.join(current_buffer)}"
                    )
                current_buffer, current_speaker = [word_text], speaker
            else:
                current_buffer.append(word_text)

        if current_buffer:  # Append the last utterance
            dialog_parts.append(
                f"{current_speaker.upper()}: {' '.join(current_buffer)}"
            )

        full_dialog = "\\n".join(dialog_parts)
        return {"words": all_words, "dialog": full_dialog}

    def _get_transcript_for_file(self, original_stereo_filepath: str):
        """Gets transcript for a stereo audio file."""
        if not self.elevenlabs_client:
            print(
                f"Skipping transcription for {original_stereo_filepath} as ElevenLabs client is not initialized."
            )
            return None

        with tempfile.TemporaryDirectory() as td:
            customer_mono_path, agent_mono_path = self._split_stereo_audio(
                original_stereo_filepath, td
            )

            if not customer_mono_path or not agent_mono_path:
                print(
                    f"Failed to split channels for {original_stereo_filepath}, skipping transcription."
                )
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_customer = executor.submit(
                    self._transcribe_channel, customer_mono_path, "customer"
                )
                future_agent = executor.submit(
                    self._transcribe_channel, agent_mono_path, "ai_agent"
                )

                words_customer = future_customer.result()
                words_agent = future_agent.result()

            if not words_customer and not words_agent:
                print(
                    f"Transcription failed for both channels of {original_stereo_filepath}."
                )
                return None

            return self._merge_and_format_transcript(words_customer, words_agent)

    def calculate_activity_windows(
        self,
        audio,
        sr,
        threshold=-35,
        min_silence_duration=0.2,
        min_activity_duration=0.1,
    ):
        """
        Calculate activity windows for each channel
        Returns lists of (start_time, end_time) tuples for each channel
        """
        # Convert threshold from dB to amplitude
        amplitude_threshold = 10 ** (threshold / 20)

        # Calculate window parameters
        # min_silence_samples = int(min_silence_duration * sr) # This variable was unused. min_silence_duration is used directly.
        min_activity_frames = int(
            min_activity_duration * sr / (0.01 * sr)
        )  # Convert min_activity_duration to frames based on hop_length

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
            current_window_start_index = None

            for i, active in enumerate(is_active):
                if active and current_window_start_index is None:
                    # Start new potential window
                    current_window_start_index = i
                elif not active and current_window_start_index is not None:
                    # Potential end of window
                    # Check if silence is long enough
                    # Look back min_silence_samples from current frame i
                    # hop_length corresponds to one frame in 'is_active' and 'times'
                    # So, min_silence_frames is min_silence_samples / hop_length
                    min_silence_frames = int(min_silence_duration / (hop_length / sr))

                    # Ensure we don't look before the start of the energy array
                    start_check_index = max(0, i - min_silence_frames)

                    # Check if all frames in the silence window are inactive
                    # And ensure the window is not at the very beginning of the segment
                    if i > 0 and not np.any(is_active[start_check_index:i]):
                        start_time = times[current_window_start_index]
                        end_time = times[i - 1]  # End time is the previous frame
                        if (end_time - start_time) >= min_activity_duration:
                            # Check if the active segment itself was long enough
                            active_segment_frames = (
                                (i - 1) - current_window_start_index + 1
                            )
                            if active_segment_frames >= min_activity_frames:
                                channel_windows.append((start_time, end_time))
                        current_window_start_index = None

            # Handle case where audio ends while potentially active
            if current_window_start_index is not None:
                start_time = times[current_window_start_index]
                end_time = times[-1]
                if (end_time - start_time) >= min_activity_duration:
                    active_segment_frames = len(is_active) - current_window_start_index
                    if active_segment_frames >= min_activity_frames:
                        channel_windows.append((start_time, end_time))

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
            for (
                agent_start,
                _agent_end,
            ) in agent_windows:  # Use _agent_end as it's not used
                # Agent started speaking while user was speaking
                if agent_start > user_start and agent_start < user_end:
                    return True

        return False

    def detect_user_interrupting_ai(self, user_windows, agent_windows):
        """Check if user interrupts AI agent during conversation"""
        for agent_window in agent_windows:
            agent_start, agent_end = agent_window

            # Check if any user window overlaps with agent window
            for user_start, _user_end in user_windows:  # Use _user_end as it's not used
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
        """Process a single audio file and calculate all metrics.
        Checks database first, if found, returns stored metrics.
        Otherwise, processes and stores new metrics."""

        db_session = next(get_db())  # Get a database session
        try:
            existing_analysis_db = get_analysis_by_filename(db_session, filename)
            if existing_analysis_db:
                print(
                    f"Found existing analysis for {filename} in database. Returning stored data."
                )
                metrics = recreate_metrics_from_db(existing_analysis_db)
                _audio, _sr, _output_path = self.downsample_audio(filename)
                metrics["downsampled_path"] = _output_path
                # If transcript is missing from old record, try to generate it now
                if not metrics.get("transcript_data") and self.elevenlabs_client:
                    print(
                        f"Transcript data missing for {filename}, attempting to generate."
                    )
                    original_file_path = os.path.join(self.input_dir, filename)
                    transcript_data = self._get_transcript_for_file(original_file_path)
                    if transcript_data:
                        metrics["transcript_data"] = transcript_data
                        # NOTE: This only updates the metrics dictionary returned.
                        # To persist this newly generated transcript for an existing record,
                        # an explicit database update operation would be needed here.
                        # For simplicity, this example assumes that if a transcript is missing,
                        # it will be generated and included in the current operation's output,
                        # but a full re-processing or a dedicated update mechanism
                        # would be required to save it back to the existing DB record.
                        print(
                            f"Generated transcript for {filename} (was missing from DB)."
                        )
                    else:
                        print(f"Failed to generate missing transcript for {filename}.")
                return metrics

            print(f"No existing analysis for {filename} in database. Processing anew.")
            # Downsample audio file (this also handles existing downsampled files)
            audio, sr, output_path = self.downsample_audio(filename)

            # Get activity windows for both channels
            activity_windows = self.calculate_activity_windows(audio, sr)

            # Right channel (index 1) is AI agent, Left channel (index 0) is user
            user_windows = activity_windows[0]  # Left channel
            agent_windows = activity_windows[1]  # Right channel

            # Get transcript
            transcript_data = None
            if self.elevenlabs_client:
                original_file_path = os.path.join(self.input_dir, filename)
                print(f"Attempting transcription for {original_file_path}...")
                transcript_data = self._get_transcript_for_file(original_file_path)
                if transcript_data:
                    print(f"Successfully transcribed {filename}.")
                else:
                    print(f"Transcription failed or yielded no data for {filename}.")
            else:
                print(
                    f"Skipping transcription for {filename} (ElevenLabs client not configured)."
                )

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
                "transcript_data": transcript_data,  # Add transcript data here
            }

            # Add new analysis to database
            add_analysis(db_session, metrics)
            print(f"Saved new analysis for {filename} to database.")
            return metrics
        finally:
            db_session.close()
