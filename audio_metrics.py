import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import traceback  # Added for error logging
import torch  # For Silero VAD
import time  # For timing the VAD processing

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
    def __init__(self, input_dir="stereo_test_calls", output_dir="sampled_test_calls", batch_only=False):
        """Initialize the calculator with input and output directories
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed audio files
            batch_only: If True, skips ElevenLabs transcription for batch processing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_only = batch_only
        self.elevenlabs_client = None
        
        # Skip ElevenLabs initialization in batch-only mode
        if not batch_only:
            # ELEVENLABS_API_KEY is now loaded from .env by load_dotenv()
            # os.getenv will retrieve it if it was successfully loaded into the environment
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            if elevenlabs_api_key:
                self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
            else:
                print(
                    "WARNING: ELEVENLABS_API_KEY not found in .env file or environment. Transcription will be skipped."
                )
        else:
            print("INFO: Running in batch-only mode. ElevenLabs transcription disabled.")

        # Silero VAD model - will be lazy loaded when needed
        self.vad_model = None
        self.sampling_rate = 16000  # Required sample rate for Silero VAD

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def downsample_audio(self, input_file, target_sr=16000):
        """Downsample stereo audio file to target sample rate and return left and right channels, preserving format."""
        # Check if input_file is an absolute path
        if os.path.isabs(input_file) and os.path.exists(input_file):
            input_path = input_file
            # Extract just the filename for output
            base_filename = os.path.basename(input_file)
        else:
            # Try relative to input directory
            input_path = os.path.join(self.input_dir, input_file)
            base_filename = input_file

        output_filename = (
            os.path.splitext(base_filename)[0]
            + "_downsampled"
            + os.path.splitext(base_filename)[1]
        )
        output_path = os.path.join(self.output_dir, output_filename)

        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Verify input file exists and is accessible
        if not os.path.exists(input_path):
            print(f"ERROR: Input file does not exist: {input_path}")
            # Try alternative paths
            alt_paths = [
                input_file,  # Try direct path without prepending input_dir
                os.path.abspath(input_file),  # Try absolute path
                os.path.join(
                    os.getcwd(), input_file
                ),  # Try relative to current working directory
            ]

            for alt_path in alt_paths:
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    print(f"Found file at alternative path: {alt_path}")
                    input_path = alt_path
                    break
            else:  # No break occurred in the for loop
                print("All path attempts failed")
                raise FileNotFoundError(
                    f"Input file not found: {input_file}. Tried paths: {input_path}, {', '.join(alt_paths)}"
                )

        # Check if downsampled file already exists
        if os.path.exists(output_path):
            print(f"Found existing downsampled file: {output_path}")
            try:
                # Ensure consistent return type with processing case
                audio, sr = librosa.load(output_path, sr=target_sr, mono=False)
                # Ensure audio is in the expected shape (channels, samples)
                if len(audio.shape) == 1:
                    audio = np.array([audio, audio])
                elif (
                    audio.shape[0] != 2 and audio.shape[1] == 2
                ):  # if (samples, channels)
                    audio = audio.T
                return audio, sr, output_path
            except Exception as e:
                print(f"ERROR: Failed to load existing downsampled file: {str(e)}")
                # Continue with re-downsampling if loading failed
                print("Will attempt to re-downsample the file...")
                print(f"Downsampling {input_file} to {output_path}")
        file_extension = os.path.splitext(input_file)[1].lower()
        try:
            if file_extension == ".mp3":
                print(f"Loading MP3 file: {input_path}")
                print(
                    f"File exists: {os.path.exists(input_path)}, size: {os.path.getsize(input_path) if os.path.exists(input_path) else 'N/A'} bytes"
                )

                # Try multiple methods to load the MP3 file
                try:
                    # Method 1: Use pydub directly
                    audio_segment = AudioSegment.from_mp3(input_path)
                    print(
                        f"Successfully loaded with pydub: channels={audio_segment.channels}, frame_rate={audio_segment.frame_rate}"
                    )
                except Exception as e1:
                    print(f"Failed to load with pydub: {str(e1)}")

                    try:
                        # Method 2: Try using librosa directly, then convert to AudioSegment
                        print("Attempting to load with librosa directly")
                        temp_audio, temp_sr = librosa.load(
                            input_path, sr=None, mono=False
                        )
                        print(
                            f"Successfully loaded with librosa: shape={temp_audio.shape}, sr={temp_sr}"
                        )

                        # Convert librosa array to AudioSegment
                        # Export to temporary WAV
                        temp_wav = os.path.join(self.output_dir, "temp_conversion.wav")
                        if len(temp_audio.shape) == 1:
                            temp_audio = np.array([temp_audio, temp_audio])
                        elif temp_audio.shape[0] != 2 and temp_audio.shape[1] == 2:
                            temp_audio = temp_audio.T

                        sf.write(temp_wav, temp_audio.T, temp_sr)

                        # Load the temp WAV with pydub
                        audio_segment = AudioSegment.from_wav(temp_wav)
                        print(
                            f"Created AudioSegment from librosa data: channels={audio_segment.channels}, frame_rate={audio_segment.frame_rate}"
                        )

                        # Clean up temp file
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    except Exception as e2:
                        print(f"Failed to load with librosa: {str(e2)}")

                        # Method 3: Try using ffmpeg directly
                        print("Attempting to convert with ffmpeg directly")
                        import subprocess

                        try:
                            # Create output directory if it doesn't exist
                            os.makedirs(self.output_dir, exist_ok=True)

                            # Use ffmpeg directly to convert
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                input_path,
                                "-ar",
                                str(target_sr),
                                "-ac",
                                "2",  # Force stereo
                                output_path,
                            ]
                            print(f"Running ffmpeg command: {' '.join(cmd)}")

                            result = subprocess.run(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                            )

                            if result.returncode != 0:
                                print(
                                    f"FFmpeg command failed with return code {result.returncode}"
                                )
                                print(f"FFmpeg stderr: {result.stderr}")
                                raise Exception(
                                    f"FFmpeg conversion failed: {result.stderr}"
                                )
                            else:
                                print("FFmpeg conversion successful")
                                # Load the converted file with librosa for return
                                audio, sr = librosa.load(
                                    output_path, sr=target_sr, mono=False
                                )
                                if len(audio.shape) == 1:
                                    audio = np.array([audio, audio])
                                elif audio.shape[0] != 2 and audio.shape[1] == 2:
                                    audio = audio.T

                                # Return early since we've already saved the output file
                                return audio, sr, output_path
                        except Exception as e3:
                            print(f"Failed to convert with ffmpeg: {str(e3)}")
                            raise Exception(
                                f"All methods failed to process audio file: {input_path}. Error 1: {str(e1)}. Error 2: {str(e2)}. Error 3: {str(e3)}"
                            )

                # If we got here, one of the methods worked and we have an audio_segment
                # Continue with normal processing

                # Ensure stereo
                if audio_segment.channels == 1:
                    print("Converting mono to stereo")
                    audio_segment = audio_segment.set_channels(2)

                # Resample if necessary
                if audio_segment.frame_rate != target_sr:
                    print(
                        f"Resampling from {audio_segment.frame_rate}Hz to {target_sr}Hz"
                    )
                    audio_segment = audio_segment.set_frame_rate(target_sr)

                print(f"Exporting to: {output_path}")
                audio_segment.export(output_path, format="mp3")
                print("Export completed successfully")

                # For consistency with librosa's output, load the downsampled mp3 with librosa
                # This is for the return value, the file is already saved.
                print("Loading downsampled file with librosa for return value")
                audio, sr = librosa.load(output_path, sr=target_sr, mono=False)
                print(f"Loaded audio shape: {audio.shape}, sr={sr}")

                # Ensure audio is in the expected shape (channels, samples)
                if len(audio.shape) == 1:  # If mono after librosa load
                    print("Converting mono to stereo array")
                    audio = np.array([audio, audio])
                elif (
                    audio.shape[0] != 2 and audio.shape[1] == 2
                ):  # if (samples, channels)
                    print("Transposing audio to get channels first")
                    audio = audio.T

                print(f"Final audio shape: {audio.shape}")
            elif file_extension == ".wav":
                print(f"Loading WAV file: {input_path}")
                print(
                    f"File exists: {os.path.exists(input_path)}, size: {os.path.getsize(input_path) if os.path.exists(input_path) else 'N/A'} bytes"
                )

                try:
                    # Try method 1: Use librosa
                    audio, sr = librosa.load(input_path, sr=None, mono=False)
                    print(
                        f"Successfully loaded with librosa: shape={audio.shape}, sr={sr}"
                    )
                except Exception as e1:
                    print(f"Failed to load with librosa: {str(e1)}")

                    try:
                        # Method 2: Try using soundfile
                        print("Attempting to load with soundfile")
                        audio, sr = sf.read(input_path)
                        # soundfile returns (samples, channels)
                        if len(audio.shape) > 1 and audio.shape[1] == 2:
                            audio = audio.T  # Convert to (channels, samples)
                        else:
                            audio = np.array([audio, audio])  # Convert mono to stereo
                        print(
                            f"Successfully loaded with soundfile: shape={audio.shape}, sr={sr}"
                        )
                    except Exception as e2:
                        print(f"Failed to load with soundfile: {str(e2)}")

                        # Method 3: Try using ffmpeg directly
                        print("Attempting to convert with ffmpeg directly")
                        import subprocess

                        try:
                            # Create output directory if it doesn't exist
                            os.makedirs(self.output_dir, exist_ok=True)

                            # Use ffmpeg directly to convert
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                input_path,
                                "-ar",
                                str(target_sr),
                                "-ac",
                                "2",  # Force stereo
                                output_path,
                            ]
                            print(f"Running ffmpeg command: {' '.join(cmd)}")

                            result = subprocess.run(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                            )

                            if result.returncode != 0:
                                print(
                                    f"FFmpeg command failed with return code {result.returncode}"
                                )
                                print(f"FFmpeg stderr: {result.stderr}")
                                raise Exception(
                                    f"FFmpeg conversion failed: {result.stderr}"
                                )
                            else:
                                print("FFmpeg conversion successful")
                                # Load the converted file with librosa for return
                                audio, sr = librosa.load(
                                    output_path, sr=target_sr, mono=False
                                )
                                if len(audio.shape) == 1:
                                    audio = np.array([audio, audio])
                                elif audio.shape[0] != 2 and audio.shape[1] == 2:
                                    audio = audio.T

                                # Return early since we've already saved the output file
                                return audio, sr, output_path
                        except Exception as e3:
                            print(f"Failed to convert with ffmpeg: {str(e3)}")
                            raise Exception(
                                f"All methods failed to process audio file: {input_path}. Error 1: {str(e1)}. Error 2: {str(e2)}. Error 3: {str(e3)}"
                            )

                # If we got here, one of the methods worked and we have audio data

                # If audio is mono, duplicate to create stereo
                if len(audio.shape) == 1:
                    print("Converting mono to stereo array")
                    audio = np.array([audio, audio])
                # Ensure audio is in the shape (channels, samples) before resample
                elif (
                    audio.shape[0] != 2 and audio.shape[1] == 2
                ):  # if (samples, channels)
                    print("Transposing audio to get channels first")
                    audio = audio.T

                # Resample if necessary
                if sr != target_sr:
                    print(f"Resampling from {sr}Hz to {target_sr}Hz")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                # Save downsampled audio
                print(f"Saving to: {output_path}")
                # soundfile.write expects (samples, channels)
                sf.write(output_path, audio.T, target_sr)
                print("File saved successfully")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            print("Downsampling completed successfully")
            return audio, target_sr, output_path

        except Exception as e:
            print(f"ERROR in downsampling: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    def _get_transcript_for_file(
        self,
        original_stereo_filepath: str,
        user_vad_segments=None,
        agent_vad_segments=None,
    ):
        """Gets transcript for a stereo audio file using ElevenLabs with static channel mapping.

        Args:
            original_stereo_filepath: Path to the stereo audio file
            user_vad_segments: VAD segments from left channel (user) for speaker correlation
            agent_vad_segments: VAD segments from right channel (agent) for speaker correlation
        """
        if not self.elevenlabs_client:
            print(
                f"Skipping transcription for {original_stereo_filepath} as ElevenLabs client is not initialized."
            )
            return None

        try:
            with open(original_stereo_filepath, "rb") as f:
                # Call ElevenLabs API with diarization disabled since we'll use static mapping
                response = self.elevenlabs_client.speech_to_text.convert(
                    file=f,
                    model_id="scribe_v1_experimental",  # Or your preferred model
                    diarize=False,  # Disable diarization - we'll use static channel mapping
                    timestamps_granularity="word",
                    tag_audio_events=False,
                )

            processed_words = []
            if hasattr(response, "words") and response.words:
                # Use static channel mapping instead of ElevenLabs diarization
                processed_words = self._map_transcript_words_to_channels(
                    response.words, user_vad_segments, agent_vad_segments
                )

                print(f"Mapped {len(processed_words)} words using static channel correlation")

            if not processed_words:
                print(
                    f"Transcription returned no words for {original_stereo_filepath}."
                )
                return None

            return self._merge_and_format_transcript(processed_words)

        except Exception as e:
            print(
                f"Error during transcription for {original_stereo_filepath}: {e}"
            )
            traceback.print_exc()
            return None

    def _merge_and_format_transcript(self, all_words_from_diarization):
        """Merges word lists (now receives a single list from diarization),
        sorts them, and formats into a dialog.
        Note: Overlap detection is now handled separately by detect_overlaps_from_vad_segments."""

        # Input 'all_words_from_diarization' should have 'speaker' attributes
        # mapped to "customer", "ai_agent" (or "other_speaker_X") by _get_transcript_for_file.
        all_words = sorted(
            all_words_from_diarization, key=lambda w: (w["start"], w["end"])
        )

        # Initialize overlap flag for each word (will be populated later if needed)
        for word in all_words:
            word["is_overlap"] = False

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
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
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
            List of dictionaries with 'start' and 'end' keys for speech segments
        """
        start_time = time.time()
        print("Starting Silero VAD speech detection...")

        # Resample to 16kHz if needed (Silero VAD requirement)
        if sr != self.sampling_rate:
            print(f"Resampling audio from {sr}Hz to {self.sampling_rate}Hz for VAD")
            audio_channel = librosa.resample(
                audio_channel, orig_sr=sr, target_sr=self.sampling_rate
            )

        # Convert to float32 tensor
        tensor_audio = torch.FloatTensor(audio_channel)

        # Get VAD model
        model, get_speech_timestamps = self.get_vad_model()

        # Get speech timestamps - more aggressive settings for better accuracy
        # Refer to https://github.com/snakers4/silero-vad/blob/0dd45f0bcd7271463c234f3bae5ad25181f9df8b/src/silero_vad/utils_vad.py#L191
        speech_timestamps = get_speech_timestamps(
            tensor_audio,
            model,
            threshold=0.72,  # Higher threshold means more aggressive voice activity detection
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=100,  # Minimum silence duration between speech chunks in ms
            min_speech_duration_ms=200,  # Minimum speech duration to be detected
            return_seconds=True,  # Get timestamps directly in seconds
        )
        """
        seconds format:
        [{'end': 2.1, 'start': 0.0},
        {'end': 4.9, 'start': 2.7},
        {'end': 6.8, 'start': 5.0}]
        about thresholds: https://github.com/snakers4/silero-vad/wiki/FAQ#which-sampling-rate-and-chunk-size-to-choose-from
        tried thresholds:
        0.5 - default, too many false positives if noise is present at customer side.
        0.7 - not bad actually, but still some false positives. this is live threshold for production.
        0.85 - too aggressive, misses some entire speech segments.
        """
        # Print raw Silero VAD timestamps for debugging
        print(
            f"Silero VAD raw timestamps for {audio_channel}: {speech_timestamps}"
        )  # Return dictionary format directly to avoid conversion issues
        # No need for any conversions as return_seconds=True already gives us float values

        end_time = time.time()
        print(
            f"Silero VAD detected {len(speech_timestamps)} speech segments in {end_time - start_time:.2f} seconds"
        )

        return speech_timestamps

    def calculate_turn_taking_latency(self, user_vad_segments, agent_vad_segments):
        """
        Calculate turn-taking latency between user utterances and agent responses
        using VAD segments. Handles overlaps by splitting them into non-overlapping turns
        and calculating latency based on actual conversation flow.

        Args:
            user_vad_segments: List of (start, end) tuples for user speech from VAD
            agent_vad_segments: List of (start, end) tuples for agent speech from VAD

        Returns:
            A tuple containing:
            - metrics (dict): Dictionary with latency stats and overlap information.
            - latency_details (list): List of detailed latency data for turn transitions.
        """
        # Convert inputs to dictionaries for consistent processing
        user_segments = self._normalize_segments(user_vad_segments, "user")
        agent_segments = self._normalize_segments(agent_vad_segments, "ai_agent")

        if not user_segments or not agent_segments:
            print("No user or agent segments to calculate latency.")
            return self._create_empty_latency_stats(), []

        # Create conversation timeline with overlap handling
        conversation_turns = self._create_overlap_aware_timeline(
            user_segments, agent_segments
        )

        # Calculate latencies between consecutive turns
        latencies, latency_details, overlap_stats = self._calculate_turn_latencies(
            conversation_turns
        )

        # Generate statistics
        latency_stats = self._generate_latency_stats(latencies)
        latency_stats.update(overlap_stats)

        return latency_stats, latency_details

    def _normalize_segments(self, segments, speaker_type):
        """Convert segments to normalized dictionary format."""
        normalized = []
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                normalized.append(
                    {"start": seg[0], "end": seg[1], "speaker": speaker_type}
                )
            elif isinstance(seg, dict) and "start" in seg and "end" in seg:
                seg_copy = seg.copy()
                seg_copy["speaker"] = speaker_type
                normalized.append(seg_copy)
            else:                
                print(f"Warning: Skipping invalid {speaker_type} segment: {seg}")
        return sorted(normalized, key=lambda x: x["start"])
    
    def _create_overlap_aware_timeline(self, user_segments, agent_segments):
        """Create timeline that properly handles overlaps"""
        all_events = []
        
        # Create start/end events for all segments
        for seg in user_segments:
            all_events.append({"time": seg["start"], "type": "start", "speaker": "user", "segment": seg})
            all_events.append({"time": seg["end"], "type": "end", "speaker": "user", "segment": seg})
        
        for seg in agent_segments:
            all_events.append({"time": seg["start"], "type": "start", "speaker": "ai_agent", "segment": seg})
            all_events.append({"time": seg["end"], "type": "end", "speaker": "ai_agent", "segment": seg})
        
        # Sort by time
        all_events.sort(key=lambda x: (x["time"], x["type"] == "start"))
        
        # Process events to create non-overlapping turns
        active_speakers = set()
        conversation_turns = []
        current_turn = None
        
        for event in all_events:
            if event["type"] == "start":
                active_speakers.add(event["speaker"])
                if current_turn is None:
                    current_turn = {
                        "start": event["time"],
                        "speaker": event["speaker"],
                        "speakers": {event["speaker"]}
                    }
                else:
                    current_turn["speakers"].add(event["speaker"])
            else:  # end event
                active_speakers.discard(event["speaker"])
                if current_turn and event["speaker"] in current_turn["speakers"]:
                    if not active_speakers:
                        # No one speaking, end current turn
                        current_turn["end"] = event["time"]
                        current_turn["speaker"] = list(current_turn["speakers"])[0] if len(current_turn["speakers"]) == 1 else "overlap"
                        conversation_turns.append(current_turn)
                        current_turn = None
        
        return conversation_turns

    def _calculate_turn_latencies(self, conversation_turns):
        """Calculate latencies between turns"""
        latencies = []
        latency_details = []
        overlap_count = 0
        
        for i in range(len(conversation_turns) - 1):
            current_turn = conversation_turns[i]
            next_turn = conversation_turns[i + 1]
            
            if current_turn["speaker"] == "user" and next_turn["speaker"] == "ai_agent":
                latency_seconds = next_turn["start"] - current_turn["end"]
                latencies.append(latency_seconds)
                
                latency_details.append({
                    "interaction_type": "user_to_agent",
                    "from_turn_end": current_turn["end"],
                    "to_turn_start": next_turn["start"],
                    "latency_seconds": latency_seconds,
                    "latency_ms": latency_seconds * 1000,
                    "rating": self.rate_latency(latency_seconds)
                })
        
        return latencies, latency_details, {"overlap_count": overlap_count}

    def _generate_latency_stats(self, latencies):
        """Generate statistical summary of latencies."""
        if not latencies:
            return {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p10_latency": 0,
                "p50_latency": 0,
                "p90_latency": 0,
            }

        # Filter out latencies below 1.5 seconds
        filtered_latencies = [lat for lat in latencies if lat >= 1.5]
        
        if not filtered_latencies:
            return {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p10_latency": 0,
                "p50_latency": 0,
                "p90_latency": 0,
            }

        return {
            "avg_latency": sum(filtered_latencies) / len(filtered_latencies),
            "min_latency": min(filtered_latencies),
            "max_latency": max(filtered_latencies),
            "p10_latency": float(np.percentile(filtered_latencies, 10))
            if len(filtered_latencies) >= 10
            else min(filtered_latencies),
            "p50_latency": float(np.percentile(filtered_latencies, 50)),
            "p90_latency": float(np.percentile(filtered_latencies, 90))
            if len(filtered_latencies) >= 10
            else max(filtered_latencies),
        }

    def _create_empty_latency_stats(self):
        """Create empty latency statistics."""
        return {
            "avg_latency": 0,
            "min_latency": 0,
            "max_latency": 0,
            "p10_latency": 0,
            "p50_latency": 0,
            "p90_latency": 0,
            "total_overlaps": 0,
            "ai_interruptions": 0,
            "user_interruptions": 0,
            "successful_handoffs": 0,
        }

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

    def calculate_talk_ratio(self, user_windows, agent_windows):
        """Calculate ratio of agent speaking time to user speaking time"""
        try:
            user_duration = sum(turn["end"] - turn["start"] for turn in user_windows)
            agent_duration = sum(turn["end"] - turn["start"] for turn in agent_windows)
        except (KeyError, TypeError, ValueError) as e:
            print(
                f"ERROR: Could not calculate durations in calculate_talk_ratio. User: {user_windows}, Agent: {agent_windows}. Error: {e}"
            )
            return float("inf")  # Or handle error appropriately

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

        # TODO: Normally I aim to find points when AI agent's voice gone wild, but
        # this could be an extreme value that gets filtered by next line. I need
        # some broken audio to test this.
        # Filter out extreme values (noise)
        valid_pitches = pitches_flat[(pitches_flat > 50) & (pitches_flat < 500)]

        if len(valid_pitches) > 0:
            return float(np.mean(valid_pitches))
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
                start = turn["start"]
                end = turn["end"]
                total_speaking_time_seconds += end - start
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                agent_speech.extend(agent_audio[start_idx:end_idx])
            except (KeyError, TypeError, ValueError) as e:
                print(
                    f"ERROR: Could not process turn in calculate_words_per_minute: {turn}. Error: {e}"
                )
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
        word_count = (
            syllable_count / 1.5
        )  # Calculate total agent speaking time in minutes
        total_speaking_time_minutes = total_speaking_time_seconds / 60

        if total_speaking_time_minutes > 0:
            return word_count / total_speaking_time_minutes
        else:
            return 0

    def _create_combined_speaker_turns(self, raw_user_segments, raw_agent_segments):
        """Combine VAD segments for user and agent into timeline of speaker turns
        
        Args:
            raw_user_segments: Raw VAD segments from user channel
            raw_agent_segments: Raw VAD segments from agent channel
            
        Returns:
            List of speaker turns with start, end, and speaker labels
        """
        user_segments = self._normalize_segments(raw_user_segments, "user")
        agent_segments = self._normalize_segments(raw_agent_segments, "ai_agent")

        if not user_segments and not agent_segments:
            return []

        conversation_turns = self._create_overlap_aware_timeline(user_segments, agent_segments)
        return conversation_turns

    def detect_overlaps_from_vad_segments(self, user_vad_segments, agent_vad_segments):
        """
        Detect overlapping speech segments directly from raw Silero VAD output.
        This provides higher granularity than using processed combined speaker turns.

        Args:
            user_vad_segments: List of dicts with {'start', 'end'} from detect_speech_silero_vad for user channel
            agent_vad_segments: List of dicts with {'start', 'end'} from detect_speech_silero_vad for agent channel

        Returns:
            Dictionary with overlap information:
            - has_user_interrupting_ai: Boolean, True if user interrupted AI
            - has_ai_interrupting_user: Boolean, True if AI interrupted user
            - total_overlap_count: Integer, total number of overlaps detected
            - overlaps: List of dicts with {'start', 'end', 'duration', 'interrupter', 'interrupted'}
        """
        print("DEBUG: Using raw VAD segments for overlap detection")
        print(f"DEBUG: User VAD segments: {len(user_vad_segments)}")
        print(f"DEBUG: Agent VAD segments: {len(agent_vad_segments)}")

        # Parameters for overlap detection
        transition_buffer = 0.2  # Reduced buffer for more sensitive detection with VAD
        min_overlap_duration = 0.05  # Minimum duration of overlap to count
        min_intrusion_ratio = 0.02  # At least 10% overlap to count as significant

        overlaps = []
        has_user_interrupting_ai = False
        has_ai_interrupting_user = False

        # Check every user segment against every agent segment for overlaps
        for user_segment in user_vad_segments:
            user_start = user_segment["start"]
            user_end = user_segment["end"]

            for agent_segment in agent_vad_segments:
                agent_start = agent_segment["start"]
                agent_end = agent_segment["end"]

                # Check if there's any overlap between these segments
                overlap_start = max(user_start, agent_start)
                overlap_end = min(user_end, agent_end)

                # Skip if no overlap
                if overlap_start >= overlap_end:
                    continue

                overlap_duration = overlap_end - overlap_start

                # Skip very short overlaps
                if overlap_duration < min_overlap_duration:
                    continue

                # Determine who started first to identify the interrupter
                if user_start < agent_start:
                    # User started first, agent is interrupting
                    interrupter = "ai_agent"
                    interrupted = "user"

                    # Check if agent started before user finished (minus buffer)
                    if agent_start < user_end - transition_buffer:
                        user_duration = user_end - user_start
                        intrusion_ratio = overlap_duration / user_duration

                        if intrusion_ratio >= min_intrusion_ratio:
                            has_ai_interrupting_user = True
                            overlaps.append(
                                {
                                    "start": overlap_start,
                                    "end": overlap_end,
                                    "duration": overlap_duration,
                                    "interrupter": interrupter,
                                    "interrupted": interrupted,
                                }
                            )

                elif agent_start < user_start:
                    # Agent started first, user is interrupting
                    interrupter = "user"
                    interrupted = "ai_agent"

                    # Check if user started before agent finished (minus buffer)
                    if user_start < agent_end - transition_buffer:
                        agent_duration = agent_end - agent_start
                        intrusion_ratio = overlap_duration / agent_duration

                        if intrusion_ratio >= min_intrusion_ratio:
                            has_user_interrupting_ai = True
                            overlaps.append(
                                {
                                    "start": overlap_start,
                                    "end": overlap_end,
                                    "duration": overlap_duration,
                                    "interrupter": interrupter,
                                    "interrupted": interrupted,
                                }
                            )

        # Remove duplicate overlaps (can happen if segments partially overlap)
        unique_overlaps = []
        for overlap in overlaps:
            is_duplicate = False
            for existing in unique_overlaps:
                # Check if this overlap is very similar to an existing one
                if (
                    abs(overlap["start"] - existing["start"]) < 0.05
                    and abs(overlap["end"] - existing["end"]) < 0.05
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_overlaps.append(overlap)

        print(f"DEBUG: Found {len(unique_overlaps)} unique overlaps from raw VAD")
        print(f"DEBUG: VAD-based overlaps: {unique_overlaps}")

        return {
            "has_user_interrupting_ai": has_user_interrupting_ai,
            "has_ai_interrupting_user": has_ai_interrupting_user,
            "total_overlap_count": len(unique_overlaps),
            "overlaps": unique_overlaps,
        }

    def _map_transcript_words_to_channels(self, transcription_words, user_vad_segments, agent_vad_segments):
        """
        Map transcript words to speakers based on static channel timing correlation.
        Uses simple overlap with channel-specific VAD segments.
        
        Args:
            transcription_words: List of word objects from ElevenLabs
            user_vad_segments: VAD segments from left channel (user) 
            agent_vad_segments: VAD segments from right channel (agent)
            
        Returns:
            List of processed words with static speaker assignments
        """
        if not user_vad_segments or not agent_vad_segments:
            print("Warning: Missing VAD segments for speaker mapping. Using default timing-based assignment.")
            
        processed_words = []
        
        for word_obj in transcription_words:
            if hasattr(word_obj, "type") and word_obj.type != "word":
                continue
                
            word_start = word_obj.start
            word_end = word_obj.end
            word_mid = (word_start + word_end) / 2
            
            # Determine speaker based on overlap with channel VAD segments
            user_overlap = self._calculate_word_channel_overlap(word_start, word_end, user_vad_segments)
            agent_overlap = self._calculate_word_channel_overlap(word_start, word_end, agent_vad_segments)
            
            # Assign to channel with highest overlap, default to agent if equal
            if user_overlap > agent_overlap:
                speaker_label = "customer"
            else:
                speaker_label = "ai_agent"
                
            word_dict = {
                "text": word_obj.text,
                "start": word_obj.start,
                "end": word_obj.end,
                "speaker": speaker_label,
                "logprob": getattr(word_obj, "logprob", None),
                "type": getattr(word_obj, "type", "word"),
            }
            processed_words.append(word_dict)
            
        return processed_words
    
    def _calculate_word_channel_overlap(self, word_start, word_end, channel_vad_segments):
        """
        Calculate overlap between a word timing and channel VAD segments.
        
        Args:
            word_start: Word start time
            word_end: Word end time  
            channel_vad_segments: VAD segments for a specific channel
            
        Returns:
            Float overlap duration
        """
        if not channel_vad_segments:
            return 0.0
            
        total_overlap = 0.0
        
        for segment in channel_vad_segments:
            if isinstance(segment, dict) and "start" in segment and "end" in segment:
                seg_start, seg_end = segment["start"], segment["end"]
            elif isinstance(segment, (list, tuple)) and len(segment) == 2:
                seg_start, seg_end = segment[0], segment[1]
            else:
                continue
                
            # Calculate overlap
            overlap_start = max(word_start, seg_start)
            overlap_end = min(word_end, seg_end)
            
            if overlap_start < overlap_end:
                total_overlap += overlap_end - overlap_start
                
        return total_overlap
    
    def _create_combined_speaker_turns(self, raw_user_segments, raw_agent_segments):
        """Combine VAD segments for user and agent into timeline of speaker turns
        
        Args:
            raw_user_segments: Raw VAD segments from user channel
            raw_agent_segments: Raw VAD segments from agent channel
            
        Returns:
            List of speaker turns with start, end, and speaker labels
        """
        user_segments = self._normalize_segments(raw_user_segments, "user")
        agent_segments = self._normalize_segments(raw_agent_segments, "ai_agent")

        if not user_segments and not agent_segments:
            return []

        conversation_turns = self._create_overlap_aware_timeline(user_segments, agent_segments)
        return conversation_turns

    def process_file(self, filename, source_url=None):
        """Process a single audio file and calculate all metrics.
        Checks database first, if found, returns stored metrics.
        Otherwise, processes and stores new metrics.

        Args:
            filename: Can be either a filename relative to the input_dir or an absolute path
            source_url: Optional URL source if file was downloaded from web
        """
        # Normalize the filename for database lookup
        base_filename = os.path.basename(filename)

        db_session = next(get_db())  # Get a database session
        try:
            existing_analysis_db = get_analysis_by_filename(db_session, base_filename)
            if existing_analysis_db:
                print(
                    f"Found existing analysis for {base_filename} in database. Returning stored data."
                )
                metrics = recreate_metrics_from_db(existing_analysis_db)
                # Ensure downsampled path is available, even if loading from DB
                _audio, _sr, _output_path = self.downsample_audio(
                    filename
                )  # audio and sr not used here
                metrics["downsampled_path"] = _output_path  # Store the path                # Only try to get transcript if ElevenLabs client is available, transcript is missing, and not in batch_only mode
                if not metrics.get("transcript_data") and self.elevenlabs_client and not self.batch_only:
                    # Attempt to get transcript if EL client available and no transcript in DB
                    print(
                        f"Transcript data missing for {filename} (from DB), attempting to generate from audio."
                    )
                    # First, we need to get VAD segments for accurate speaker correlation
                    audio, sr, _ = self.downsample_audio(filename)
                    print(
                        "Processing user channel with Silero VAD for transcript correlation..."
                    )
                    raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)
                    print(
                        "Processing agent channel with Silero VAD for transcript correlation..."
                    )
                    raw_agent_vad_segments = self.detect_speech_silero_vad(audio[1], sr)

                    original_file_path = os.path.join(self.input_dir, filename)
                    transcript_data = self._get_transcript_for_file(
                        original_file_path,
                        raw_user_vad_segments,
                        raw_agent_vad_segments,
                    )
                    if transcript_data:
                        metrics["transcript_data"] = transcript_data
                        print(
                            f"Generated transcript for {filename} (was missing from DB)."
                        )
                    else:
                        print(
                            f"Failed to generate transcript for {filename}. No transcript data available."
                        )
                return metrics
            print(
                f"No existing analysis for {base_filename} in database. Processing anew."
            )
            # Downsample audio file (this also handles existing downsampled files)
            audio, sr, output_path = self.downsample_audio(
                filename
            )  # Process user channel with Silero VAD for more accurate speech detection
            print("Processing user channel with Silero VAD...")
            raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)

            # Process agent channel with Silero VAD
            print("Processing agent channel with Silero VAD...")
            raw_agent_vad_segments = self.detect_speech_silero_vad(
                audio[1], sr
            )  # Get transcript with improved overlap handling (if ElevenLabs client is available)
            # Now we pass the VAD segments for accurate speaker correlation
            transcript_data = None
            if self.elevenlabs_client:
                # Check if filename is already an absolute path
                if os.path.isabs(filename) and os.path.exists(filename):
                    original_file_path = filename
                else:
                    original_file_path = os.path.join(self.input_dir, filename)
                print(f"Attempting transcription for {original_file_path}...")
                transcript_data = self._get_transcript_for_file(
                    original_file_path, raw_user_vad_segments, raw_agent_vad_segments
                )
                if transcript_data:
                    print(f"Successfully transcribed {filename}.")
                    print(
                        f"Found {transcript_data.get('overlap_count', 0)} overlapping speech segments."
                    )
                else:
                    print(f"Transcription failed or yielded no data for {filename}.")
            else:
                if self.batch_only:
                    print(f"Skipping transcription for {filename} (running in batch-only mode).")
                else:
                    print(f"Skipping transcription for {filename} (ElevenLabs client not configured).")

            # Create combined and merged speaker turns
            print("Creating combined speaker turns...")
            combined_speaker_turns = self._create_combined_speaker_turns(
                raw_user_vad_segments, raw_agent_vad_segments
            )  # Extract merged turns for each speaker for subsequent calculations
            user_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in combined_speaker_turns
                if turn["speaker"] == "user"
            ]
            agent_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in combined_speaker_turns
                if turn["speaker"] == "ai_agent"
            ]

            print(
                f"Combined into {len(user_speech_turns)} user turns and {len(agent_speech_turns)} agent turns."
            )

            # Calculate VAD-based turn-taking latency metrics using merged turns (Agent_Start - User_End)
            vad_latency_metrics, vad_latency_details = (
                self.calculate_turn_taking_latency(
                    user_speech_turns, agent_speech_turns
                )
            )  # vad_latency_metrics now includes 'ai_interruptions_handled_in_latency'
            # Calculate overlap detection using raw VAD segments for higher granularity
            overlap_data = self.detect_overlaps_from_vad_segments(
                raw_user_vad_segments, raw_agent_vad_segments
            )  # Calculate metrics
            metrics = {
                "filename": base_filename,  # Use base filename for consistent database lookups
                "source_url": source_url,  # Store source URL if downloaded from web
                "original_path": filename,  # Store original path for reference
                "downsampled_path": output_path,
                "combined_speaker_turns": combined_speaker_turns,  # Store the combined turns
                # Use merged turns for all relevant metrics
                "user_vad_segments": user_speech_turns,
                "agent_vad_segments": agent_speech_turns,
                # New VAD-based latency metrics (Agent VAD Start - User VAD End)
                "vad_latency_metrics": vad_latency_metrics,
                "vad_latency_details": vad_latency_details,
                # Overlap detection from combined speaker turns
                "overlap_data": overlap_data,
                "ai_interrupting_user": overlap_data["has_ai_interrupting_user"],
                "user_interrupting_ai": overlap_data["has_user_interrupting_ai"],
                # Other metrics
                "talk_ratio": self.calculate_talk_ratio(
                    user_speech_turns, agent_speech_turns
                ),
                "average_pitch": self.calculate_average_pitch(
                    audio, sr
                ),  # Pitch still uses raw audio
                "words_per_minute": self.calculate_words_per_minute(
                    audio, sr, agent_speech_turns
                ),
                "transcript_data": transcript_data,
            }
            # The "no transcript data available" message will be shown when transcript fails
            add_analysis(db_session, metrics)
            print(
                f"Saved new analysis for {base_filename} to database with improved transcript handling."
            )
            return metrics
        finally:
            db_session.close()

    def _map_transcript_words_to_channels(self, transcription_words, user_vad_segments, agent_vad_segments):
        """
        Map transcript words to speakers based on static channel timing correlation.
        Uses simple overlap with channel-specific VAD segments.
        
        Args:
            transcription_words: List of word objects from ElevenLabs
            user_vad_segments: VAD segments from left channel (user) 
            agent_vad_segments: VAD segments from right channel (agent)
            
        Returns:
            List of processed words with static speaker assignments
        """
        if not user_vad_segments or not agent_vad_segments:
            print("Warning: Missing VAD segments for speaker mapping. Using default timing-based assignment.")
            
        processed_words = []
        
        for word_obj in transcription_words:
            if hasattr(word_obj, "type") and word_obj.type != "word":
                continue
                
            word_start = word_obj.start
            word_end = word_obj.end
            word_mid = (word_start + word_end) / 2
            
            # Determine speaker based on overlap with channel VAD segments
            user_overlap = self._calculate_word_channel_overlap(word_start, word_end, user_vad_segments)
            agent_overlap = self._calculate_word_channel_overlap(word_start, word_end, agent_vad_segments)
            
            # Assign to channel with highest overlap, default to agent if equal
            if user_overlap > agent_overlap:
                speaker_label = "customer"
            else:
                speaker_label = "ai_agent"
                
            word_dict = {
                "text": word_obj.text,
                "start": word_obj.start,
                "end": word_obj.end,
                "speaker": speaker_label,
                "logprob": getattr(word_obj, "logprob", None),
                "type": getattr(word_obj, "type", "word"),
            }
            processed_words.append(word_dict)
            
        return processed_words
    
    def _calculate_word_channel_overlap(self, word_start, word_end, channel_vad_segments):
        """
        Calculate overlap between a word timing and channel VAD segments.
        
        Args:
            word_start: Word start time
            word_end: Word end time  
            channel_vad_segments: VAD segments for a specific channel
            
        Returns:
            Float overlap duration
        """
        if not channel_vad_segments:
            return 0.0
            
        total_overlap = 0.0
        
        for segment in channel_vad_segments:
            if isinstance(segment, dict) and "start" in segment and "end" in segment:
                seg_start, seg_end = segment["start"], segment["end"]
            elif isinstance(segment, (list, tuple)) and len(segment) == 2:
                seg_start, seg_end = segment[0], segment[1]
            else:
                continue
                
            # Calculate overlap
            overlap_start = max(word_start, seg_start)
            overlap_end = min(word_end, seg_end)
            
            if overlap_start < overlap_end:
                total_overlap += overlap_end - overlap_start
                
        return total_overlap
    
    def _create_combined_speaker_turns(self, raw_user_segments, raw_agent_segments):
        """Combine VAD segments for user and agent into timeline of speaker turns
        
        Args:
            raw_user_segments: Raw VAD segments from user channel
            raw_agent_segments: Raw VAD segments from agent channel
            
        Returns:
            List of speaker turns with start, end, and speaker labels
        """
        user_segments = self._normalize_segments(raw_user_segments, "user")
        agent_segments = self._normalize_segments(raw_agent_segments, "ai_agent")

        if not user_segments and not agent_segments:
            return []

        conversation_turns = self._create_overlap_aware_timeline(user_segments, agent_segments)
        return conversation_turns

    def process_file(self, filename, source_url=None):
        """Process a single audio file and calculate all metrics.
        Checks database first, if found, returns stored metrics.
        Otherwise, processes and stores new metrics.

        Args:
            filename: Can be either a filename relative to the input_dir or an absolute path
            source_url: Optional URL source if file was downloaded from web
        """
        # Normalize the filename for database lookup
        base_filename = os.path.basename(filename)

        db_session = next(get_db())  # Get a database session
        try:
            existing_analysis_db = get_analysis_by_filename(db_session, base_filename)
            if existing_analysis_db:
                print(
                    f"Found existing analysis for {base_filename} in database. Returning stored data."
                )
                metrics = recreate_metrics_from_db(existing_analysis_db)
                # Ensure downsampled path is available, even if loading from DB
                _audio, _sr, _output_path = self.downsample_audio(
                    filename
                )  # audio and sr not used here
                metrics["downsampled_path"] = _output_path  # Store the path                # Only try to get transcript if ElevenLabs client is available, transcript is missing, and not in batch_only mode
                if not metrics.get("transcript_data") and self.elevenlabs_client and not self.batch_only:
                    # Attempt to get transcript if EL client available and no transcript in DB
                    print(
                        f"Transcript data missing for {filename} (from DB), attempting to generate from audio."
                    )
                    # First, we need to get VAD segments for accurate speaker correlation
                    audio, sr, _ = self.downsample_audio(filename)
                    print(
                        "Processing user channel with Silero VAD for transcript correlation..."
                    )
                    raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)
                    print(
                        "Processing agent channel with Silero VAD for transcript correlation..."
                    )
                    raw_agent_vad_segments = self.detect_speech_silero_vad(audio[1], sr)

                    original_file_path = os.path.join(self.input_dir, filename)
                    transcript_data = self._get_transcript_for_file(
                        original_file_path,
                        raw_user_vad_segments,
                        raw_agent_vad_segments,
                    )
                    if transcript_data:
                        metrics["transcript_data"] = transcript_data
                        print(
                            f"Generated transcript for {filename} (was missing from DB)."
                        )
                    else:
                        print(
                            f"Failed to generate transcript for {filename}. No transcript data available."
                        )
                return metrics
            print(
                f"No existing analysis for {base_filename} in database. Processing anew."
            )
            # Downsample audio file (this also handles existing downsampled files)
            audio, sr, output_path = self.downsample_audio(
                filename
            )  # Process user channel with Silero VAD for more accurate speech detection
            print("Processing user channel with Silero VAD...")
            raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)

            # Process agent channel with Silero VAD
            print("Processing agent channel with Silero VAD...")
            raw_agent_vad_segments = self.detect_speech_silero_vad(
                audio[1], sr
            )  # Get transcript with improved overlap handling (if ElevenLabs client is available)
            # Now we pass the VAD segments for accurate speaker correlation
            transcript_data = None
            if self.elevenlabs_client:
                # Check if filename is already an absolute path
                if os.path.isabs(filename) and os.path.exists(filename):
                    original_file_path = filename
                else:
                    original_file_path = os.path.join(self.input_dir, filename)
                print(f"Attempting transcription for {original_file_path}...")
                transcript_data = self._get_transcript_for_file(
                    original_file_path, raw_user_vad_segments, raw_agent_vad_segments
                )
                if transcript_data:
                    print(f"Successfully transcribed {filename}.")
                    print(
                        f"Found {transcript_data.get('overlap_count', 0)} overlapping speech segments."
                    )
                else:
                    print(f"Transcription failed or yielded no data for {filename}.")
            else:
                if self.batch_only:
                    print(f"Skipping transcription for {filename} (running in batch-only mode).")
                else:
                    print(f"Skipping transcription for {filename} (ElevenLabs client not configured).")

            # Create combined and merged speaker turns
            print("Creating combined speaker turns...")
            combined_speaker_turns = self._create_combined_speaker_turns(
                raw_user_vad_segments, raw_agent_vad_segments
            )  # Extract merged turns for each speaker for subsequent calculations
            user_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in combined_speaker_turns
                if turn["speaker"] == "user"
            ]
            agent_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in combined_speaker_turns
                if turn["speaker"] == "ai_agent"
            ]

            print(
                f"Combined into {len(user_speech_turns)} user turns and {len(agent_speech_turns)} agent turns."
            )

            # Calculate VAD-based turn-taking latency metrics using merged turns (Agent_Start - User_End)
            vad_latency_metrics, vad_latency_details = (
                self.calculate_turn_taking_latency(
                    user_speech_turns, agent_speech_turns
                )
            )  # vad_latency_metrics now includes 'ai_interruptions_handled_in_latency'
            # Calculate overlap detection using raw VAD segments for higher granularity
            overlap_data = self.detect_overlaps_from_vad_segments(
                raw_user_vad_segments, raw_agent_vad_segments
            )  # Calculate metrics
            metrics = {
                "filename": base_filename,  # Use base filename for consistent database lookups
                "source_url": source_url,  # Store source URL if downloaded from web
                "original_path": filename,  # Store original path for reference
                "downsampled_path": output_path,
                "combined_speaker_turns": combined_speaker_turns,  # Store the combined turns
                # Use merged turns for all relevant metrics
                "user_vad_segments": user_speech_turns,
                "agent_vad_segments": agent_speech_turns,
                # New VAD-based latency metrics (Agent VAD Start - User VAD End)
                "vad_latency_metrics": vad_latency_metrics,
                "vad_latency_details": vad_latency_details,
                # Overlap detection from combined speaker turns
                "overlap_data": overlap_data,
                "ai_interrupting_user": overlap_data["has_ai_interrupting_user"],
                "user_interrupting_ai": overlap_data["has_user_interrupting_ai"],
                # Other metrics
                "talk_ratio": self.calculate_talk_ratio(
                    user_speech_turns, agent_speech_turns
                ),
                "average_pitch": self.calculate_average_pitch(
                    audio, sr
                ),  # Pitch still uses raw audio
                "words_per_minute": self.calculate_words_per_minute(
                    audio, sr, agent_speech_turns
                ),
                "transcript_data": transcript_data,
            }
            # The "no transcript data available" message will be shown when transcript fails
            add_analysis(db_session, metrics)
            print(
                f"Saved new analysis for {base_filename} to database with improved transcript handling."
            )
            return metrics
        finally:
            db_session.close()
