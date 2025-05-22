import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import traceback  # Added for error logging
import torch  # For Silero VAD
import time     # For timing the VAD processing

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

        # Silero VAD model - will be lazy loaded when needed
        self.vad_model = None
        self.sampling_rate = 16000  # Required sample rate for Silero VAD

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

    def get_vad_model(self):
        """Lazy load the Silero VAD model to save resources if it's not used"""
        if self.vad_model is None:
            print("Loading Silero VAD model...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            (get_speech_timestamps, _, _, _, _) = utils
            
            self.vad_model = model
            self.get_speech_timestamps = get_speech_timestamps
            print("Silero VAD model loaded successfully")
        
        return self.vad_model, self.get_speech_timestamps
        
    def detect_speech_silero_vad(self, audio_channel, sr):
        """
        Detect speech segments using Silero VAD for more accurate speech detection
        
        Args:
            audio_channel: Audio data for a single channel
            sr: Sample rate of the audio
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        start_time = time.time()
        print("Starting Silero VAD speech detection...")
        
        # Resample to 16kHz if needed (Silero VAD requirement)
        if sr != self.sampling_rate:
            print(f"Resampling audio from {sr}Hz to {self.sampling_rate}Hz for VAD")
            audio_channel = librosa.resample(audio_channel, orig_sr=sr, target_sr=self.sampling_rate)
        
        # Convert to float32 tensor
        tensor_audio = torch.FloatTensor(audio_channel)
        
        # Get VAD model
        model, get_speech_timestamps = self.get_vad_model()
        
        # Get speech timestamps - more aggressive settings for better accuracy
        speech_timestamps = get_speech_timestamps(
            tensor_audio,
            model,
            threshold=0.5,  # Higher threshold means more aggressive voice activity detection
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=400,  # Minimum silence duration between speech chunks in ms
            min_speech_duration_ms=200,   # Minimum speech duration to be detected
            window_size_samples=512       # Window size for processing
        )
        
        # Print raw Silero VAD timestamps for debugging
        print(f"Raw Silero VAD timestamps: {speech_timestamps}")
        
        # Convert timestamps to seconds
        speech_segments = []
        for segment in speech_timestamps:
            start = segment['start'] / self.sampling_rate
            end = segment['end'] / self.sampling_rate
            speech_segments.append((start, end))
        
        end_time = time.time()
        print(f"Silero VAD detected {len(speech_segments)} speech segments in {end_time - start_time:.2f} seconds")
        
        return speech_segments    
    def calculate_turn_taking_latency(self, user_vad_segments, agent_vad_segments):
        """
        Calculate turn-taking latency between user utterances and agent responses
        using VAD for user speech and activity windows for agent.
        Each user utterance is paired with the next available agent response.
        Handles AI interruptions by skipping the interrupted user turn and the interrupting AI turn.
        
        Args:
            user_vad_segments: List of {'start': float, 'end': float} for user speech from VAD
            agent_vad_segments: List of {'start': float, 'end': float} for agent speech from VAD
            
        Returns:
            A tuple containing:
            - metrics (dict): Dictionary with latency stats and ai_interruptions_handled_in_latency.
            - latency_details (list): List of detailed latency data for non-interrupted turns.
        """
        latencies = []
        latency_details = []
        ai_interruptions_handled_in_latency = 0

        if not user_vad_segments or not agent_vad_segments:
            print("No user VAD segments or agent windows to calculate latency.")
            return {
                "avg_latency": 0, "min_latency": 0, "max_latency": 0,
                "p10_latency": 0, "p50_latency": 0, "p90_latency": 0,
                "ai_interruptions_handled_in_latency": 0
            }, []
        
        # Ensure segments are sorted by start time
        user_vad_segments = sorted(user_vad_segments, key=lambda x: float(x['start']))
        
        min_agent_speech_duration = 0  # VAD should handle meaningful duration
        significant_agent_windows = sorted(
            [w for w in agent_vad_segments if (float(w['end']) - float(w['start'])) >= min_agent_speech_duration],
            key=lambda x: float(x['start'])
        )

        if not significant_agent_windows:
            print("No significant agent windows to calculate latency against.")
            return {
                "avg_latency": 0, "min_latency": 0, "max_latency": 0,
                "p10_latency": 0, "p50_latency": 0, "p90_latency": 0,
                "ai_interruptions_handled_in_latency": 0
            }, []

        print(f"Initial counts - User VAD segments: {len(user_vad_segments)}, Significant Agent VAD segments: {len(significant_agent_windows)}")

        current_agent_search_start_idx = 0

        for user_segment in user_vad_segments:
            user_start = float(user_segment['start'])
            user_end = float(user_segment['end'])
            
            processed_this_user_segment = False

            for agent_search_idx in range(current_agent_search_start_idx, len(significant_agent_windows)):
                agent_window = significant_agent_windows[agent_search_idx]
                agent_start = float(agent_window['start'])
                # agent_end = float(agent_window['end']) # Not needed for this logic step

                # Check for AI interruption: AI starts speaking during the user's current speech segment.
                if user_start < agent_start < user_end:
                    ai_interruptions_handled_in_latency += 1
                    print(f"AI interruption detected and handled for latency: User {user_segment} interrupted by Agent {agent_window}")
                    # This user segment and this agent window are skipped for latency.
                    # Advance the agent search index past the interrupting agent window for the next user segment.
                    current_agent_search_start_idx = agent_search_idx + 1
                    processed_this_user_segment = True
                    break # Move to the next user_segment

                # Check for standard latency: AI starts after the user has finished.
                elif agent_start > user_end:
                    latency_seconds = agent_start - user_end
                    # Latency must be positive.
                    if latency_seconds > 0.001: # Adding a small threshold to avoid zero/negative due to float precision
                        latencies.append(latency_seconds)
                        latency_details.append({
                            "latency_seconds": latency_seconds,
                            "latency_ms": latency_seconds * 1000,
                            "start_time": agent_start, 
                            "user_end": user_end,
                            "agent_start": agent_start,
                            "user_turn_details": user_segment,
                            "agent_turn_details": agent_window,
                            "rating": self.rate_latency(latency_seconds)
                        })
                    # This agent window is now "consumed" as a response.
                    current_agent_search_start_idx = agent_search_idx + 1
                    processed_this_user_segment = True
                    break # Move to the next user_segment
                
                # Else (agent_start <= user_start or agent_start == user_end):
                # This agent window is too early for the current user segment (e.g., started before user)
                # or started exactly when user ended (zero latency, typically handled by VAD merging or min_latency threshold).
                # Continue searching for a suitable agent window for the *current* user_segment.
                # current_agent_search_start_idx is NOT advanced here, as this agent window was not "used".
                # However, if all subsequent agent windows are also "too early" for this user segment,
                # this user segment won't get a latency pair.
                # The crucial part is that current_agent_search_start_idx only advances when an agent window is consumed.

            if not processed_this_user_segment and current_agent_search_start_idx < len(significant_agent_windows):
                # If the user segment was not processed (no interruption, no response found starting AFTER it),
                # and there are still agent windows that were earlier than this user segment or started concurrently,
                # we need to ensure current_agent_search_start_idx is at least at the first agent window
                # that could potentially respond to the *next* user segment.
                # This situation arises if all remaining agent windows start before or during the current user segment ends,
                # but none qualify as an interruption as defined (user_start < agent_start < user_end).
                # We need to advance current_agent_search_start_idx past those agent windows that definitely won't apply to future user turns
                # if they occurred before the current user turn ended.
                temp_advance_idx = current_agent_search_start_idx
                while temp_advance_idx < len(significant_agent_windows) and \
                      float(significant_agent_windows[temp_advance_idx]['start']) <= user_end:
                    temp_advance_idx += 1
                current_agent_search_start_idx = temp_advance_idx


        print(f"Calculated {len(latencies)} turn-taking latencies. Detected and handled {ai_interruptions_handled_in_latency} AI interruptions in latency context.")

        if not latencies:
            latency_stats = {
                "avg_latency": 0, "min_latency": 0, "max_latency": 0,
                "p10_latency": 0, "p50_latency": 0, "p90_latency": 0,
            }
        else:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p10_latency = np.percentile(latencies, 10) if len(latencies) >= 10 else min_latency
            p50_latency = np.percentile(latencies, 50)
            p90_latency = np.percentile(latencies, 90) if len(latencies) >= 10 else max_latency
            latency_stats = {
                "avg_latency": avg_latency, "min_latency": min_latency, "max_latency": max_latency,
                "p10_latency": p10_latency, "p50_latency": p50_latency, "p90_latency": p90_latency,
            }
        
        latency_stats["ai_interruptions_handled_in_latency"] = ai_interruptions_handled_in_latency
        
        return latency_stats, latency_details

    def rate_latency(self, latency_seconds):
        """
        Rate the latency according to specified thresholds
        
        Args:
            latency_seconds: Latency in seconds
            
        Returns:
            String rating of the latency: Perfect, Good, OK, Bad, Poor
        """
        if latency_seconds < 2:
            return "Perfect"
        elif latency_seconds < 3:
            return "Good"
        elif latency_seconds < 4:
            return "OK"
        elif latency_seconds < 5:
            return "Bad"
        else:
            return "Poor"

    def detect_ai_interrupting_user(self, user_windows, agent_windows):
        """Check if AI agent interrupts user during conversation, using improved detection logic"""
        # Use same parameters as in transcript overlap detection for consistency
        transition_buffer = 0.2  # 200ms buffer to allow for natural transitions
        min_overlap_duration = 0.15  # Minimum duration of overlap to count
        min_intrusion_ratio = 0.20  # At least 20% overlap to count as significant

        for user_turn in user_windows:
            try:
                user_start = float(user_turn['start'])
                user_end = float(user_turn['end'])
            except (KeyError, TypeError, ValueError) as e:
                print(f"ERROR: Could not process user_turn: {user_turn}. Error: {e}")
                continue  # Skip this problematic turn

            user_duration = user_end - user_start

            # Skip very short utterances that might be noise
            if user_duration < 0.3:  # Less than 300ms
                continue

            # Check if any agent window starts while user is speaking (with a buffer)
            for agent_turn in agent_windows:
                try:
                    agent_start = float(agent_turn['start'])
                    agent_end = float(agent_turn['end'])
                except (KeyError, TypeError, ValueError) as e:
                    print(f"ERROR: Could not process agent_turn: {agent_turn}. Error: {e}")
                    continue # Skip this problematic turn
                
                # Agent started speaking while user was speaking
                if (
                    agent_start > user_start
                    and agent_start < user_end - transition_buffer
                ):
                    # Calculate how much of the user\'s speech was interrupted
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

        for agent_turn in agent_windows:
            try:
                agent_start = float(agent_turn['start'])
                agent_end = float(agent_turn['end'])
            except (KeyError, TypeError, ValueError) as e:
                print(f"ERROR: Could not process agent_turn: {agent_turn}. Error: {e}")
                continue # Skip this problematic turn

            agent_duration = agent_end - agent_start

            # Skip very short utterances that might be noise
            if agent_duration < 0.3:  # Less than 300ms
                continue

            # Check if any user window starts while agent is speaking (with a buffer)
            for user_turn in user_windows:
                try:
                    user_start = float(user_turn['start'])
                    user_end = float(user_turn['end'])
                except (KeyError, TypeError, ValueError) as e:
                    print(f"ERROR: Could not process user_turn: {user_turn}. Error: {e}")
                    continue # Skip this problematic turn

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
        try:
            user_duration = sum(float(turn['end']) - float(turn['start']) for turn in user_windows)
            agent_duration = sum(float(turn['end']) - float(turn['start']) for turn in agent_windows)
        except (KeyError, TypeError, ValueError) as e:
            print(f"ERROR: Could not calculate durations in calculate_talk_ratio. User: {user_windows}, Agent: {agent_windows}. Error: {e}")
            return float("inf") # Or handle error appropriately

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
        total_speaking_time_seconds = 0
        for turn in agent_windows:
            try:
                start = float(turn['start'])
                end = float(turn['end'])
                total_speaking_time_seconds += (end - start)
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                agent_speech.extend(agent_audio[start_idx:end_idx])
            except (KeyError, TypeError, ValueError) as e:
                print(f"ERROR: Could not process turn in calculate_words_per_minute: {turn}. Error: {e}")
                continue

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
        total_speaking_time_minutes = total_speaking_time_seconds / 60

        if total_speaking_time_minutes > 0:
            return word_count / total_speaking_time_minutes
        else:
            return 0

    def _generate_placeholder_transcript_from_segments(
        self, user_vad_segments, agent_vad_segments
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
        for i, (start, end) in enumerate(user_vad_segments):
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

        for i, (start, end) in enumerate(agent_vad_segments):
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
        all_segments = [(start, end, "CUSTOMER") for start, end in user_vad_segments] + [
            (start, end, "AI_AGENT") for start, end in agent_vad_segments
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

    def _create_combined_speaker_turns(self, raw_user_segments, raw_agent_segments): # Removed max_merge_gap
        """
        Combines and merges VAD segments for user and agent into a single timeline of speaker turns.
        Segments are merged if they are from the same speaker and are continuous or overlapping.
        The final list of turns is sorted by start time.

        Args:
            raw_user_segments: List of (start, end) tuples for user VAD.
            raw_agent_segments: List of (start, end) tuples for agent VAD.

        Returns:
            A list of dictionaries, where each dictionary is a turn with
            {'speaker': 'user'/'ai_agent', 'start': float, 'end': float}, sorted by start time.
        """
        tagged_segments = []
        # Ensure segments are actual tuples/lists of two numbers before processing
        for seg in raw_user_segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                tagged_segments.append({'speaker': 'user', 'start': seg[0], 'end': seg[1]})
            else:
                print(f"Warning: Skipping invalid user segment: {seg}")
        for seg in raw_agent_segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                tagged_segments.append({'speaker': 'ai_agent', 'start': seg[0], 'end': seg[1]})
            else:
                print(f"Warning: Skipping invalid agent segment: {seg}")

        if not tagged_segments:
            return []

        # Sort all segments by start time to process them chronologically
        tagged_segments.sort(key=lambda x: x['end'])
        print(f"DEBUG: Sorted segments: {tagged_segments}")
        merged_turns = []
        if not tagged_segments: # Should be caught by the check above, but good for safety
            return merged_turns

        # Initialize with the first segment
        merged_turns.append(tagged_segments[0].copy())

        for i in range(1, len(tagged_segments)):
            current_segment = tagged_segments[i]
            last_merged_turn = merged_turns[-1]

            # Check if the current segment can be merged with the last merged turn:
            # 1. Same speaker
            # 2. Current segment starts at or before the last merged turn ends (continuous or overlapping)
            if current_segment['speaker'] == last_merged_turn['speaker']:
                # Merge: extend the end time of the last_merged_turn
                print(f"Merging segments: {last_merged_turn} with {current_segment}")
                last_merged_turn['end'] = max(last_merged_turn['end'], current_segment['end'])
            else:
                # Different speaker or a gap, so start a new turn
                merged_turns.append(current_segment.copy())

        merged_turns.sort(key=lambda x: x['start'])  # Sort by start time
        # Pretty print the merged turns for debugging
        for turn in merged_turns:
            print(f"DEBUG: Merged turn: {turn}")
        # The merged_turns list is already sorted by start time due to the initial sort and processing order.
        print(f"Combined {len(raw_user_segments) + len(raw_agent_segments)} raw segments into {len(merged_turns)} merged speaker turns.")
        return merged_turns

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
                # Ensure downsampled path is available, even if loading from DB
                _audio, _sr, _output_path = self.downsample_audio(filename) # audio and sr not used here
                metrics["downsampled_path"] = _output_path


                # If loading from DB, we might need to regenerate placeholder from new VAD segments if they exist
                # This part assumes that if 'user_vad_segments' and 'agent_vad_segments' are in metrics,
                # they are the ones to be used.
                user_speech_turns_for_placeholder = metrics.get("user_vad_segments", [])
                agent_speech_turns_for_placeholder = metrics.get("agent_vad_segments", [])
                
                # Convert to list of tuples if they are list of dicts, for _generate_placeholder_transcript_from_segments
                if user_speech_turns_for_placeholder and isinstance(user_speech_turns_for_placeholder[0], dict):
                    user_speech_turns_for_placeholder = [(t['start'], t['end']) for t in user_speech_turns_for_placeholder]
                if agent_speech_turns_for_placeholder and isinstance(agent_speech_turns_for_placeholder[0], dict):
                    agent_speech_turns_for_placeholder = [(t['start'], t['end']) for t in agent_speech_turns_for_placeholder]


                if not metrics.get("transcript_data") and self.elevenlabs_client is None: # Only if no EL and no transcript
                    if user_speech_turns_for_placeholder or agent_speech_turns_for_placeholder:
                        print(
                            f"Generating placeholder transcript from VAD segments for {filename} (from DB)."
                        )
                        transcript_data = (
                            self._generate_placeholder_transcript_from_segments(
                                user_speech_turns_for_placeholder, agent_speech_turns_for_placeholder
                            )
                        )
                        metrics["transcript_data"] = transcript_data
                elif not metrics.get("transcript_data") and self.elevenlabs_client:
                     # Attempt to get transcript if EL client available and no transcript in DB
                    print(
                        f"Transcript data missing for {filename} (from DB), attempting to generate from audio."
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
                        # Fallback to placeholder if EL fails
                        if user_speech_turns_for_placeholder or agent_speech_turns_for_placeholder:
                            print(
                                f"EL transcription failed for {filename} (from DB), generating placeholder."
                            )
                            transcript_data = (
                                self._generate_placeholder_transcript_from_segments(
                                    user_speech_turns_for_placeholder, agent_speech_turns_for_placeholder
                                )
                            )
                            metrics["transcript_data"] = transcript_data
                        else:
                            print(
                                f"Failed to generate missing transcript for {filename} and no VAD segments for placeholder."
                            )
                return metrics

            print(f"No existing analysis for {filename} in database. Processing anew.")
            # Downsample audio file (this also handles existing downsampled files)
            audio, sr, output_path = self.downsample_audio(filename)

            # Get transcript with improved overlap handling (if ElevenLabs client is available)
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
            
            # Process user channel with Silero VAD for more accurate speech detection
            print("Processing user channel with Silero VAD...")
            raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)
            
            # Process agent channel with Silero VAD
            print("Processing agent channel with Silero VAD...")
            raw_agent_vad_segments = self.detect_speech_silero_vad(audio[1], sr)
            
            # Create combined and merged speaker turns
            print("Creating combined speaker turns...")
            combined_speaker_turns = self._create_combined_speaker_turns(raw_user_vad_segments, raw_agent_vad_segments)
            
            # Extract merged turns for each speaker for subsequent calculations
            user_speech_turns = [{'start': float(turn['start']), 'end': float(turn['end'])} for turn in combined_speaker_turns if turn['speaker'] == 'user']
            agent_speech_turns = [{'start': float(turn['start']), 'end': float(turn['end'])} for turn in combined_speaker_turns if turn['speaker'] == 'ai_agent']
            
            print(f"Combined into {len(user_speech_turns)} user turns and {len(agent_speech_turns)} agent turns.")

            # Calculate VAD-based turn-taking latency metrics using merged turns (Agent_Start - User_End)
            vad_latency_metrics, vad_latency_details = self.calculate_turn_taking_latency(
                user_speech_turns, 
                agent_speech_turns 
            )
            # vad_latency_metrics now includes 'ai_interruptions_handled_in_latency'

            # Calculate metrics
            metrics = {
                "filename": filename,
                "downsampled_path": output_path,
                "combined_speaker_turns": combined_speaker_turns, # Store the combined turns
                # Use merged turns for all relevant metrics
                "user_vad_segments": user_speech_turns, 
                "agent_vad_segments": agent_speech_turns,
                
                # New VAD-based latency metrics (Agent VAD Start - User VAD End)
                "vad_latency_metrics": vad_latency_metrics,
                "vad_latency_details": vad_latency_details,
                # Other metrics (now also VAD-based for agent, using merged turns)
                "ai_interrupting_user": self.detect_ai_interrupting_user(
                    user_speech_turns, agent_speech_turns
                ),
                "user_interrupting_ai": self.detect_user_interrupting_ai(
                    user_speech_turns, agent_speech_turns
                ),
                "talk_ratio": self.calculate_talk_ratio(user_speech_turns, agent_speech_turns),
                "average_pitch": self.calculate_average_pitch(audio, sr), # Pitch still uses raw audio
                "words_per_minute": self.calculate_words_per_minute(
                    audio, sr, agent_speech_turns 
                ),
                "transcript_data": transcript_data, 
            }

            # If placeholder transcript generation is needed (e.g. EL failed or not used)
            if not metrics["transcript_data"] and (user_speech_turns or agent_speech_turns):
                 print(f"Generating placeholder transcript using merged VAD-based speaker turns for {filename}")
                 # _generate_placeholder_transcript_from_segments expects lists of (start,end) tuples
                 # user_speech_turns and agent_speech_turns are lists of dicts {'start':s, 'end':e}
                 # So, convert them back for this specific function if its signature is not changed.
                 # OR, update _generate_placeholder_transcript_from_segments to accept list of dicts.
                 # For now, let's assume _generate_placeholder_transcript_from_segments is flexible or we adapt.
                 # Let's adapt the call to match its expected (start,end) tuple list format:
                 user_tuples_for_placeholder = [(turn['start'], turn['end']) for turn in user_speech_turns]
                 agent_tuples_for_placeholder = [(turn['start'], turn['end']) for turn in agent_speech_turns]
                 
                 placeholder_transcript = self._generate_placeholder_transcript_from_segments(
                     user_tuples_for_placeholder, agent_tuples_for_placeholder
                 )
                 metrics["transcript_data"] = placeholder_transcript


            # Add new analysis to database
            add_analysis(db_session, metrics)
            print(
                f"Saved new analysis for {filename} to database with improved transcript handling."
            )
            return metrics
        finally:
            db_session.close()
