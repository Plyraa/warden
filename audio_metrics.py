import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import traceback  # Added for error logging

# Database imports
from database import (
    get_db,
    add_analysis,
    get_analysis_by_filename,
    recreate_metrics_from_db,
)

# Load environment variables from .env file
from dotenv import load_dotenv

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

    def _get_transcript_for_file(self, original_stereo_filepath: str):
        """Gets transcript for a stereo audio file using ElevenLabs diarization."""
        if not self.elevenlabs_client:
            print(
                f"Skipping transcription for {original_stereo_filepath} as ElevenLabs client is not initialized."
            )
            return None

        try:
            with open(original_stereo_filepath, "rb") as f:
                # Call ElevenLabs API with diarization enabled
                response = self.elevenlabs_client.speech_to_text.convert(
                    file=f,
                    model_id="scribe_v1_experimental",  # Or your preferred model
                    diarize=True,  # Enable diarization
                    num_speakers=2,  # Specify number of speakers
                    timestamps_granularity="word",
                    tag_audio_events=False,
                )

            processed_words = []
            if hasattr(response, "words") and response.words:
                # Speaker mapping:
                # Based on testing, ElevenLabs uses "speaker_0" and "speaker_1".
                # User confirmed that AI (agent) speaks first and is "speaker_0".
                # Customer is therefore "speaker_1".
                speaker_map = {
                    "speaker_0": "ai_agent",  # AI agent speaks first
                    "speaker_1": "customer",
                }

                for word_obj in response.words:
                    if hasattr(word_obj, "type") and word_obj.type != "word":
                        continue

                    raw_speaker_id = getattr(word_obj, "speaker_id", None)

                    if raw_speaker_id not in speaker_map:
                        # Handle unexpected speaker IDs, or if speaker_id attribute is missing
                        print(
                            f"Warning: Unexpected or missing speaker_id: {raw_speaker_id} for word '{word_obj.text}'"
                        )
                        # Assign a generic label or skip, depending on desired robustness
                        mapped_speaker_label = (
                            f"unknown_{raw_speaker_id or 'nospeakerid'}"
                        )
                    else:
                        mapped_speaker_label = speaker_map[raw_speaker_id]

                    word_dict = {
                        "text": word_obj.text,
                        "start": word_obj.start,
                        "end": word_obj.end,
                        "speaker": mapped_speaker_label,  # Use the mapped label
                        "logprob": getattr(word_obj, "logprob", None),
                        "type": getattr(word_obj, "type", "word"),
                        "raw_speaker_id": raw_speaker_id,
                    }
                    processed_words.append(word_dict)

            if not processed_words:
                print(
                    f"Transcription returned no words for {original_stereo_filepath}."
                )
                return None

            return self._merge_and_format_transcript(processed_words)

        except Exception as e:
            print(
                f"Error during diarized transcription for {original_stereo_filepath}: {e}"
            )
            traceback.print_exc()
            return None

    def _merge_and_format_transcript(self, all_words_from_diarization):
        """Merges word lists (now receives a single list from diarization),
        sorts them, and formats into a dialog.
        Also detects and marks overlapping speech segments."""

        # Input 'all_words_from_diarization' should have 'speaker' attributes
        # mapped to "customer", "ai_agent" (or "other_speaker_X") by _get_transcript_for_file.
        all_words = sorted(
            all_words_from_diarization, key=lambda w: (w["start"], w["end"])
        )

        for word in all_words:
            word["is_overlap"] = False

        transition_buffer = 0.2
        min_overlap_duration = 0.15
        min_intrusion_ratio = 0.20

        # Filter for known speaker roles for overlap detection
        customer_words = [w for w in all_words if w["speaker"] == "customer"]
        agent_words = [w for w in all_words if w["speaker"] == "ai_agent"]

        def detect_overlaps(words_a, words_b):
            def group_into_phrases(words):
                if not words:
                    return []
                phrases = []
                current_phrase = [words[0]]
                for i in range(1, len(words)):
                    if words[i]["start"] - words[i - 1]["end"] < 0.3:
                        current_phrase.append(words[i])
                    else:
                        if current_phrase:
                            start = current_phrase[0]["start"]
                            end = current_phrase[-1]["end"]
                            phrases.append((start, end, current_phrase))
                        current_phrase = [words[i]]
                if current_phrase:
                    start = current_phrase[0]["start"]
                    end = current_phrase[-1]["end"]
                    phrases.append((start, end, current_phrase))
                return phrases

            phrases_a = group_into_phrases(words_a)
            phrases_b = group_into_phrases(words_b)

            for start_a, end_a, phrase_words_a in phrases_a:
                duration_a = end_a - start_a
                for start_b, end_b, phrase_words_b in phrases_b:
                    if start_b < end_a and start_a < end_b:
                        overlap_start = max(start_a, start_b)
                        overlap_end = min(end_a, end_b)
                        overlap_duration = overlap_end - overlap_start
                        if overlap_duration > min_overlap_duration:
                            if start_b > start_a:
                                if start_b < end_a - transition_buffer:
                                    intrusion_ratio = overlap_duration / duration_a
                                    if intrusion_ratio > min_intrusion_ratio:
                                        for word in phrase_words_a:
                                            if word["end"] > overlap_start:
                                                word["is_overlap"] = True
                                        for word in phrase_words_b:
                                            if word["start"] < overlap_end:
                                                word["is_overlap"] = True
                            else:
                                duration_b = end_b - start_b
                                if start_a < end_b - transition_buffer:
                                    intrusion_ratio = overlap_duration / duration_b
                                    if intrusion_ratio > min_intrusion_ratio:
                                        for word in phrase_words_b:
                                            if word["end"] > overlap_start:
                                                word["is_overlap"] = True
                                        for word in phrase_words_a:
                                            if word["start"] < overlap_end:
                                                word["is_overlap"] = True

        # Only detect overlaps if both customer and agent words are present
        if customer_words and agent_words:
            detect_overlaps(customer_words, agent_words)

        dialog_parts = []
        current_buffer = []
        current_speaker = None

        for word_info in all_words:
            word_text = word_info["text"]
            speaker = word_info["speaker"]
            is_overlap = word_info.get("is_overlap", False)
            marked_text = f"{word_text}[OVERLAP]" if is_overlap else word_text

            if speaker != current_speaker:
                if current_buffer and current_speaker:
                    dialog_parts.append(
                        f"{str(current_speaker).upper()}: {' '.join(current_buffer)}"
                    )
                current_buffer = [marked_text]
                current_speaker = speaker
            else:
                current_buffer.append(marked_text)

        if current_buffer and current_speaker:
            dialog_parts.append(
                f"{str(current_speaker).upper()}: {' '.join(current_buffer)}"
            )

        full_dialog = "\\n".join(dialog_parts)

        return {
            "words": all_words,
            "dialog": full_dialog,
            "has_overlaps": any(word.get("is_overlap", False) for word in all_words),
            "overlap_count": sum(
                1 for word in all_words if word.get("is_overlap", False)
            ),
        }

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
        """Check if AI agent interrupts user during conversation, using improved detection logic"""
        # Use same parameters as in transcript overlap detection for consistency
        transition_buffer = 0.2  # 200ms buffer to allow for natural transitions
        min_overlap_duration = 0.15  # Minimum duration of overlap to count
        min_intrusion_ratio = 0.20  # At least 20% overlap to count as significant

        for user_start, user_end in user_windows:
            user_duration = user_end - user_start

            # Skip very short utterances that might be noise
            if user_duration < 0.3:  # Less than 300ms
                continue

            # Check if any agent window starts while user is speaking (with a buffer)
            for agent_start, agent_end in agent_windows:
                # Agent started speaking while user was speaking
                if (
                    agent_start > user_start
                    and agent_start < user_end - transition_buffer
                ):
                    # Calculate how much of the user's speech was interrupted
                    overlap_duration = min(user_end, agent_end) - agent_start

                    # Only count as interruption if substantial
                    if (
                        overlap_duration > min_overlap_duration
                        and overlap_duration / user_duration > min_intrusion_ratio
                    ):
                        return True

        return False

    def detect_user_interrupting_ai(self, user_windows, agent_windows):
        """Check if user interrupts AI agent during conversation, using improved detection logic"""
        # Use same parameters as in transcript overlap detection for consistency
        transition_buffer = 0.2  # 200ms buffer to allow for natural transitions
        min_overlap_duration = 0.15  # Minimum duration of overlap to count
        min_intrusion_ratio = 0.20  # At least 20% overlap to count as significant

        for agent_start, agent_end in agent_windows:
            agent_duration = agent_end - agent_start

            # Skip very short utterances that might be noise
            if agent_duration < 0.3:  # Less than 300ms
                continue

            # Check if any user window starts while agent is speaking (with a buffer)
            for user_start, user_end in user_windows:
                # User started speaking while agent was speaking
                if (
                    user_start > agent_start
                    and user_start < agent_end - transition_buffer
                ):
                    # Calculate how much of the agent's speech was interrupted
                    overlap_duration = min(agent_end, user_end) - user_start

                    # Only count as interruption if substantial
                    if (
                        overlap_duration > min_overlap_duration
                        and overlap_duration / agent_duration > min_intrusion_ratio
                    ):
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

    def _generate_placeholder_transcript_from_segments(
        self, user_windows, agent_windows
    ):
        """
        Generate a placeholder transcript from speech segments when actual transcript is not available.
        This helps visualize the conversation flow without re-calling the ElevenLabs API.

        Args:
            user_windows: List of (start_time, end_time) tuples for user speech
            agent_windows: List of (start_time, end_time) tuples for agent speech

        Returns:
            Dictionary with transcript data in the same format as from _merge_and_format_transcript
        """
        words_customer = []
        words_agent = []

        # Create word-level data from speech segments with estimated word positions
        for i, (start, end) in enumerate(user_windows):
            # Create a simple placeholder word for each segment
            duration = end - start
            word_count = max(
                1, int(duration / 0.3)
            )  # Rough estimate: 1 word per 0.3 seconds

            for j in range(word_count):
                # Distribute words evenly within the segment
                word_start = start + (duration * j / word_count)
                word_end = start + (duration * (j + 1) / word_count)

                words_customer.append(
                    {
                        "text": "...",  # Using ellipsis as placeholder for better visualization
                        "start": word_start,
                        "end": word_end,
                        "speaker": "customer",
                    }
                )

        for i, (start, end) in enumerate(agent_windows):
            # Create a simple placeholder word for each segment
            duration = end - start
            word_count = max(
                1, int(duration / 0.3)
            )  # Rough estimate: 1 word per 0.3 seconds

            for j in range(word_count):
                # Distribute words evenly within the segment
                word_start = start + (duration * j / word_count)
                word_end = start + (duration * (j + 1) / word_count)

                words_agent.append(
                    {
                        "text": "...",  # Using ellipsis as placeholder for better visualization
                        "start": word_start,
                        "end": word_end,
                        "speaker": "ai_agent",
                    }
                )

        # Merge the words from both speakers
        transcript_data = self._merge_and_format_transcript(
            words_customer + words_agent
        )

        # Mark that this transcript was generated from segments
        transcript_data["generated_from_segments"] = True

        # Create a more readable dialog representation
        dialog_parts = []

        # Sort segments by start time to get conversation flow
        all_segments = [(start, end, "CUSTOMER") for start, end in user_windows] + [
            (start, end, "AI_AGENT") for start, end in agent_windows
        ]
        all_segments.sort(key=lambda x: x[0])  # Sort by start time

        # Build a simplified dialog representation
        for start, end, speaker in all_segments:
            duration = end - start
            seconds = round(duration, 1)
            dialog_parts.append(
                f"{speaker}: [Speech segment from {start:.1f}s to {end:.1f}s, duration {seconds}s]"
            )

        # Replace the placeholder dialog with this more informative version
        transcript_data["dialog"] = "\\n".join(dialog_parts)

        return transcript_data

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

                # Check if we need to generate transcript data
                if not metrics.get("transcript_data") and self.elevenlabs_client:
                    # Check if we have speech segments that can be used instead of calling the API
                    user_windows = metrics.get("user_windows", [])
                    agent_windows = metrics.get("agent_windows", [])

                    if user_windows and agent_windows:
                        print(
                            f"Found speech segments for {filename} in database. Generating transcript from existing segments."
                        )
                        # Create mock transcript data from speech segments without calling ElevenLabs API
                        transcript_data = (
                            self._generate_placeholder_transcript_from_segments(
                                user_windows, agent_windows
                            )
                        )
                        metrics["transcript_data"] = transcript_data
                        print(
                            f"Generated placeholder transcript from speech segments for {filename}."
                        )
                    else:
                        # No segments available, attempt transcription if client is available
                        print(
                            f"Transcript data missing for {filename}, attempting to generate from audio."
                        )
                        original_file_path = os.path.join(self.input_dir, filename)
                        transcript_data = self._get_transcript_for_file(
                            original_file_path
                        )
                        if transcript_data:
                            metrics["transcript_data"] = transcript_data
                            print(
                                f"Generated transcript for {filename} (was missing from DB)."
                            )
                        else:
                            print(
                                f"Failed to generate missing transcript for {filename}."
                            )

                return metrics

            print(f"No existing analysis for {filename} in database. Processing anew.")
            # Downsample audio file (this also handles existing downsampled files)
            audio, sr, output_path = self.downsample_audio(filename)

            # Get activity windows for both channels
            activity_windows = self.calculate_activity_windows(audio, sr)

            # Right channel (index 1) is AI agent, Left channel (index 0) is user
            user_windows = activity_windows[0]  # Left channel
            agent_windows = activity_windows[1]  # Right channel

            # Get transcript with improved overlap handling
            transcript_data = None
            if self.elevenlabs_client:
                original_file_path = os.path.join(self.input_dir, filename)
                print(f"Attempting transcription for {original_file_path}...")
                transcript_data = self._get_transcript_for_file(original_file_path)
                if transcript_data:
                    print(f"Successfully transcribed {filename}.")
                    print(
                        f"Found {transcript_data.get('overlap_count', 0)} overlapping speech segments."
                    )
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
                "transcript_data": transcript_data,  # Add enhanced transcript data here
            }

            # Add new analysis to database
            add_analysis(db_session, metrics)
            print(
                f"Saved new analysis for {filename} to database with improved transcript handling."
            )
            return metrics
        finally:
            db_session.close()
