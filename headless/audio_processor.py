"""
Streamlined audio processor for headless Warden - no database, no ElevenLabs
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import traceback
from pathlib import Path
import torch
import time
import constants
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import find_peaks
from noise_reduction import apply_noise_reduction

class AudioProcessor:
    def __init__(self, audio_dir: Path):
        self.sampling_rate = 16000
        self.vad_model = None
        self.get_speech_timestamps = None
        self.audio_dir = audio_dir

    def get_vad_model(self):
        """Get Silero VAD model and utility functions"""
        if self.vad_model is None:
            try:
                # Load model and utilities from torch hub
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model = model
                # Extract the get_speech_timestamps function from utils
                self.get_speech_timestamps = utils[0]  # get_speech_timestamps is the first utility
                print("Silero VAD model loaded successfully")
                return self.vad_model, self.get_speech_timestamps
            except Exception as e:
                print(f"Error loading Silero VAD model: {e}")
                raise
        else:
            return self.vad_model, self.get_speech_timestamps

    def downsample_audio(self, input_path, target_sr=16000):
        """Downsample stereo audio file to target sample rate and return left and right channels"""
        # Check if input_file is an absolute path

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        base_filename = os.path.basename(input_path)

        output_filename = (
            os.path.splitext(base_filename)[0] + "_downsampled" + os.path.splitext(base_filename)[1]
        )
        output_path = os.path.join(self.audio_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Removing existing downsampled file for fresh processing: {output_path}")
            os.remove(output_path)

        file_extension = os.path.splitext(input_path)[1].lower()
        
        try:
            if file_extension == ".mp3":
                print(f"Loading MP3 file: {input_path}")
                # Use pydub for MP3 files
                audio_segment = AudioSegment.from_mp3(input_path)
                
                # Ensure stereo
                if audio_segment.channels == 1:
                    print("Converting mono to stereo")
                    audio_segment = audio_segment.set_channels(2)

                # Resample if necessary
                if audio_segment.frame_rate != target_sr:
                    print(f"Resampling from {audio_segment.frame_rate}Hz to {target_sr}Hz")
                    audio_segment = audio_segment.set_frame_rate(target_sr)

                print(f"Exporting to: {output_path}")
                audio_segment.export(output_path, format="mp3")
                
                # Load with librosa for return value
                audio, sr = librosa.load(output_path, sr=target_sr, mono=False)
                if len(audio.shape) == 1:
                    audio = np.array([audio, audio])
                elif audio.shape[0] != 2 and audio.shape[1] == 2:
                    audio = audio.T

            elif file_extension == ".wav":
                print(f"Loading WAV file: {input_path}")
                audio, sr = librosa.load(input_path, sr=None, mono=False)
                
                # If audio is mono, duplicate to create stereo
                if len(audio.shape) == 1:
                    audio = np.array([audio, audio])
                elif audio.shape[0] != 2 and audio.shape[1] == 2:
                    audio = audio.T

                # Resample if necessary
                if sr != target_sr:
                    print(f"Resampling from {sr}Hz to {target_sr}Hz")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                # Save downsampled audio
                print(f"Saving to: {output_path}")
                sf.write(output_path, audio.T, target_sr)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            return audio, target_sr, output_path

        except Exception as e:
            print(f"ERROR in downsampling: {str(e)}")
            raise

    def detect_speech_silero_vad(self, audio_channel, sr):
        """Detect speech segments using Silero VAD"""
        start_time = time.time()
        print("Starting Silero VAD speech detection...")

        # Resample to 16kHz if needed
        if sr != self.sampling_rate:
            print(f"Resampling audio from {sr}Hz to {self.sampling_rate}Hz for VAD")
            audio_channel = librosa.resample(
                audio_channel, orig_sr=sr, target_sr=self.sampling_rate
            )

        # Convert to float32 tensor
        tensor_audio = torch.FloatTensor(audio_channel)

        # Get VAD model and function
        model, get_speech_timestamps_func = self.get_vad_model()

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps_func(
            tensor_audio,
            model,
            threshold=0.7,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=100,
            min_speech_duration_ms=100,
            return_seconds=True,
        )

        end_time = time.time()
        print(f"Silero VAD detected {len(speech_timestamps)} speech segments in {end_time - start_time:.2f} seconds")

        return speech_timestamps

    def calculate_turn_taking_latency(self, user_vad_segments, agent_vad_segments):
        """Calculate turn-taking latency between user utterances and agent responses"""
        user_segments = self._normalize_segments(user_vad_segments, "user")
        agent_segments = self._normalize_segments(agent_vad_segments, "ai_agent")

        if not user_segments or not agent_segments:
            print("No user or agent segments to calculate latency.")
            return self._create_empty_latency_stats(), []

        # Create conversation timeline with overlap handling
        conversation_turns = self._create_overlap_aware_timeline(user_segments, agent_segments)

        # Calculate latencies between consecutive turns
        latencies, latency_details, overlap_stats = self._calculate_turn_latencies(conversation_turns)

        # Generate statistics
        latency_stats = self._generate_latency_stats(latencies)
        latency_stats.update(overlap_stats)

        return latency_stats, latency_details

    def _normalize_segments(self, segments, speaker_type):
        """Convert segments to normalized dictionary format"""
        normalized = []
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                normalized.append({"start": seg[0], "end": seg[1], "speaker": speaker_type})
            elif isinstance(seg, dict) and "start" in seg and "end" in seg:
                seg_copy = seg.copy()
                seg_copy["speaker"] = speaker_type
                normalized.append(seg_copy)
            else:
                print(f"Warning: Skipping invalid {speaker_type} segment: {seg}")
        return sorted(normalized, key=lambda x: x["start"])

    def _create_overlap_aware_timeline(self, user_segments, agent_segments, max_gap=1.5):
        """
        Create a detailed timeline that properly handles overlaps like audio_metrics.py.
        This creates a timeline where overlaps are explicitly marked but original segments are preserved.
        """
        # Combine all segments with timestamps
        all_segments = []
        for seg in user_segments:
            all_segments.append({
                "start": seg["start"], 
                "end": seg["end"], 
                "speaker": seg["speaker"],
                "original": True
            })
        for seg in agent_segments:
            all_segments.append({
                "start": seg["start"], 
                "end": seg["end"], 
                "speaker": seg["speaker"],
                "original": True
            })
        
        all_segments.sort(key=lambda x: x["start"])
        
        if not all_segments:
            return []
        
        # Create a detailed timeline by finding all time boundaries
        time_points = set()
        for seg in all_segments:
            time_points.add(seg["start"])
            time_points.add(seg["end"])
        
        time_points = sorted(time_points)
        
        # Create timeline segments for each time interval
        timeline_segments = []
        
        for i in range(len(time_points) - 1):
            start_time = time_points[i]
            end_time = time_points[i + 1]
            
            # Find which speakers are active in this interval
            active_speakers = []
            for seg in all_segments:
                if seg["start"] <= start_time and seg["end"] >= end_time:
                    active_speakers.append(seg["speaker"])
            
            if active_speakers:
                if len(active_speakers) == 1:
                    # Single speaker
                    timeline_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "speaker": active_speakers[0],
                        "is_overlap": False
                    })
                else:
                    # Multiple speakers - overlap
                    timeline_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "speaker": "overlap",
                        "speakers": active_speakers,
                        "is_overlap": True
                    })
        
        # Now merge adjacent segments from same speaker (non-overlaps only)
        if not timeline_segments:
            return timeline_segments
        
        merged_segments = []
        current_segment = timeline_segments[0].copy()
        
        for i in range(1, len(timeline_segments)):
            next_segment = timeline_segments[i]
            
            # Merge if same speaker, not overlaps, and close enough
            if (current_segment["speaker"] == next_segment["speaker"] and 
                not current_segment.get("is_overlap") and 
                not next_segment.get("is_overlap") and
                next_segment["start"] - current_segment["end"] <= max_gap):
                # Extend current segment
                current_segment["end"] = next_segment["end"]
            else:
                # Different speaker or overlap, start new segment
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Don't forget the last segment
        merged_segments.append(current_segment)

        # Add debug output to track timeline creation
        print(f"DEBUG: Created {len(merged_segments)} timeline segments:")
        #for i, seg in enumerate(merged_segments):
            #if seg.get("is_overlap"):
                #speakers_str = ', '.join(seg.get("speakers", []))
                #print(f"DEBUG: Turn {i+1}: {seg['speaker']} {seg['start']:.1f}-{seg['end']:.1f} (OVERLAP: [{speakers_str}])")
            #else:
                #print(f"DEBUG: Turn {i+1}: {seg['speaker']} {seg['start']:.1f}-{seg['end']:.1f}")
        
        return merged_segments

    def _calculate_turn_latencies(self, conversation_turns):
        """Calculate latencies between conversational turns, looking back to find
        the actual end of the previous speaker's conversational segment."""
        latencies = []
        latency_details = []
        overlap_count = 0
        processed_transitions = set()  # Track processed transitions to avoid duplicates
        
        # Find speaker transitions and calculate true conversational latencies
        i = 0
        while i < len(conversation_turns):
            current_turn = conversation_turns[i]
            
            # Skip overlaps (matching audio_metrics.py behavior)
            if current_turn.get("is_overlap"):
                i += 1
                continue
            
            # Look ahead to find the next different speaker
            next_different_speaker_turn = None
            j = i + 1
            while j < len(conversation_turns):
                candidate_turn = conversation_turns[j]
                
                # Skip overlaps
                if candidate_turn.get("is_overlap"):
                    j += 1
                    continue
                
                # Found a different speaker
                if candidate_turn["speaker"] != current_turn["speaker"]:
                    next_different_speaker_turn = candidate_turn
                    break
                    
                j += 1
            
            # If we found a speaker transition, calculate latency
            if next_different_speaker_turn:
                # Find the FIRST segment of the current speaker's conversational turn
                # Look backwards to find where this speaker's turn started
                first_segment_end = current_turn["end"]
                first_segment_start = current_turn["start"]
                k = i
                while k >= 0:
                    check_turn = conversation_turns[k]
                    if (check_turn["speaker"] == current_turn["speaker"] and 
                        not check_turn.get("is_overlap")):
                        first_segment_end = check_turn["end"]
                        first_segment_start = check_turn["start"]
                        # Continue looking backwards to find the very first segment
                        if k > 0:
                            prev_check = conversation_turns[k-1]
                            # If previous turn is different speaker or overlap, we found the start
                            if (prev_check["speaker"] != current_turn["speaker"] or 
                                prev_check.get("is_overlap")):
                                break
                    else:
                        # Found a different speaker or overlap, stop looking
                        break
                    k -= 1
                
                # Create a unique key for this transition to avoid duplicates
                transition_key = (first_segment_start, next_different_speaker_turn["start"])
                
                # Only process if we haven't seen this exact transition before
                if transition_key not in processed_transitions:
                    processed_transitions.add(transition_key)
                    
                    # Only calculate user_to_agent latencies (same as original logic)
                    if current_turn["speaker"] == "user" and next_different_speaker_turn["speaker"] == "ai_agent":
                        # Calculate latency from FIRST segment end of user to start of agent
                        latency_seconds = next_different_speaker_turn["start"] - first_segment_end
                        
                        # Debug output to show what we're calculating
                        print(f"DEBUG: Checking latency from user turn {i} (FIRST segment end: {first_segment_end:.1f}s) to agent turn (start: {next_different_speaker_turn['start']:.1f}s)")
                        
                        # Only include positive latencies (actual response delays) - matching audio_metrics.py
                        if latency_seconds > 0.001:  # Small threshold for floating point precision
                            latencies.append(latency_seconds)
                            
                            # Debug output to help verify calculations
                            print(f"DEBUG: Found latency: {latency_seconds:.2f}s from user FIRST segment end at {first_segment_end:.1f}s to agent start at {next_different_speaker_turn['start']:.1f}s")
                            
                            latency_details.append({
                                "interaction_type": "user_to_agent",
                                "from_turn_end": first_segment_end,
                                "to_turn_start": next_different_speaker_turn["start"],
                                "latency_seconds": latency_seconds,
                                "latency_ms": latency_seconds * 1000
                            })
                        else:
                            print(f"DEBUG: Skipping negative/zero latency: {latency_seconds:.2f}s")
            
            i += 1
        
        return latencies, latency_details, {"overlap_count": overlap_count}

    def _generate_latency_stats(self, latencies):
        """Generate latency statistics matching audio_metrics.py approach"""
        # Filter out latencies below 1.5 seconds (matching audio_metrics.py)
        filtered_latencies = [latency for latency in latencies if latency > 1.5]
        
        if not filtered_latencies:
            return {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p50_latency": 0,
                "p90_latency": 0,
            }

        return {
            "avg_latency": sum(filtered_latencies) / len(filtered_latencies),
            "min_latency": min(filtered_latencies),
            "max_latency": max(filtered_latencies),
            "p50_latency": float(np.percentile(filtered_latencies, 50)),
            "p90_latency": float(np.percentile(filtered_latencies, 90))
            if len(filtered_latencies) >= 4
            else max(filtered_latencies),
        }

    def _create_empty_latency_stats(self):
        """Create empty latency statistics"""
        return {
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p90_latency": 0.0,
            "min_latency": 0.0,
            "max_latency": 0.0
        }

    def detect_overlaps_from_vad_segments(self, user_vad_segments, agent_vad_segments):
        """Detect overlapping speech segments from VAD output"""
        transition_buffer = 0.2
        min_overlap_duration = 0.05
        min_intrusion_ratio = 0.02

        overlaps = []
        has_user_interrupting_ai = False
        has_ai_interrupting_user = False

        for user_segment in user_vad_segments:
            user_start = user_segment["start"]
            user_end = user_segment["end"]

            for agent_segment in agent_vad_segments:
                agent_start = agent_segment["start"]
                agent_end = agent_segment["end"]

                # Check for overlap
                overlap_start = max(user_start, agent_start)
                overlap_end = min(user_end, agent_end)

                if overlap_start >= overlap_end:
                    continue

                overlap_duration = overlap_end - overlap_start

                if overlap_duration < min_overlap_duration:
                    continue

                # Determine interrupter
                if user_start < agent_start:
                    interrupter = "ai_agent"
                    interrupted = "user"
                    if agent_start < user_end - transition_buffer:
                        user_duration = user_end - user_start
                        intrusion_ratio = overlap_duration / user_duration
                        if intrusion_ratio >= min_intrusion_ratio:
                            has_ai_interrupting_user = True
                            overlaps.append({
                                "start": overlap_start,
                                "end": overlap_end,
                                "duration": overlap_duration,
                                "interrupter": interrupter,
                                "interrupted": interrupted,
                            })

                elif agent_start < user_start:
                    interrupter = "user"
                    interrupted = "ai_agent"
                    if user_start < agent_end - transition_buffer:
                        agent_duration = agent_end - agent_start
                        intrusion_ratio = overlap_duration / agent_duration
                        if intrusion_ratio >= min_intrusion_ratio:
                            has_user_interrupting_ai = True
                            overlaps.append({
                                "start": overlap_start,
                                "end": overlap_end,
                                "duration": overlap_duration,
                                "interrupter": interrupter,
                                "interrupted": interrupted,
                            })

        # Remove duplicates
        unique_overlaps = []
        for overlap in overlaps:
            is_duplicate = False
            for existing in unique_overlaps:
                if (abs(overlap["start"] - existing["start"]) < 0.05 and 
                    abs(overlap["end"] - existing["end"]) < 0.05):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_overlaps.append(overlap)

        return {
            "has_user_interrupting_ai": has_user_interrupting_ai,
            "has_ai_interrupting_user": has_ai_interrupting_user,
            "total_overlap_count": len(unique_overlaps),
            "overlaps": unique_overlaps,
        }

    def calculate_talk_ratio(self, user_windows, agent_windows):
        """Calculate ratio of agent speaking time to user speaking time"""
        try:
            user_duration = sum(turn["end"] - turn["start"] for turn in user_windows)
            agent_duration = sum(turn["end"] - turn["start"] for turn in agent_windows)
        except (KeyError, TypeError, ValueError):
            return float("inf")

        if user_duration == 0:
            return float("inf")

        return agent_duration / user_duration

    def calculate_average_pitch(self, audio, sr):
        """Calculate average pitch for agent channel (right channel)"""
        agent_audio = audio[1]

        # Calculate pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=agent_audio, sr=sr)
        pitches_flat = pitches[magnitudes > np.median(magnitudes)]
        valid_pitches = pitches_flat[(pitches_flat > 50) & (pitches_flat < 500)]

        if len(valid_pitches) > 0:
            return float(np.mean(valid_pitches))
        else:
            return 0

    def calculate_words_per_minute(self, audio, sr, agent_windows):
        """Estimate words per minute for agent speech"""
        agent_audio = audio[1]
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
            except (KeyError, TypeError, ValueError):
                continue

        if not agent_speech:
            return 0

        agent_speech = np.array(agent_speech)

        # Calculate RMS energy
        hop_length = int(0.01 * sr)
        frame_length = int(0.025 * sr)
        energy = librosa.feature.rms(y=agent_speech, frame_length=frame_length, hop_length=hop_length)[0]

        # Find peaks in energy (syllables)
        
        peaks, _ = find_peaks(energy, distance=5, prominence=0.01)

        # Estimate words
        syllable_count = len(peaks)
        word_count = syllable_count / 1.5
        total_speaking_time_minutes = total_speaking_time_seconds / 60

        if total_speaking_time_minutes > 0:
            return word_count / total_speaking_time_minutes
        else:
            return 0
    def combine_segments_by_speaker(self, user_vad_segments, agent_vad_segments, max_gap=1.5):
        """
        Combine VAD segments by speaker with gap merging, similar to audio_metrics.py
        
        Args:
            user_vad_segments: List of user VAD segments
            agent_vad_segments: List of agent VAD segments  
            max_gap: Maximum gap in seconds to merge segments from same speaker
            
        Returns:
            List of combined speaker turns with speaker labels
        """
        all_segments = []
        
        # Add speaker labels
        for seg in user_vad_segments:
            segment_copy = seg.copy() if isinstance(seg, dict) else {"start": seg[0], "end": seg[1]}
            segment_copy["speaker"] = "user"
            all_segments.append(segment_copy)
            
        for seg in agent_vad_segments:
            segment_copy = seg.copy() if isinstance(seg, dict) else {"start": seg[0], "end": seg[1]}
            segment_copy["speaker"] = "ai_agent"
            all_segments.append(segment_copy)
        
        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])
        
        if not all_segments:
            return []
        
        # Merge segments from same speaker that are close together
        combined_segments = []
        current_segment = all_segments[0].copy()
        
        for i in range(1, len(all_segments)):
            next_segment = all_segments[i]
            
            # If same speaker and gap is small enough, merge
            if (current_segment["speaker"] == next_segment["speaker"] and 
                next_segment["start"] - current_segment["end"] <= max_gap):
                # Extend current segment
                current_segment["end"] = max(current_segment["end"], next_segment["end"])
            else:
                # Different speaker or gap too large, start new segment
                combined_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Don't forget the last segment
        combined_segments.append(current_segment)
        
        return combined_segments

    def process_file(self, filename):
        """Process a single audio file and calculate all metrics"""
        try:
            print(f"Processing file: {filename}")
            
            # Downsample audio file
            audio, sr, output_path = self.downsample_audio(filename)

            # Apply noise reduction to left channel (user channel) only
            print("Applying noise reduction to user channel...")
            try:
                audio = apply_noise_reduction(audio, sr)
                print("Noise reduction completed successfully")
            except Exception as e:
                print(f"Warning: Noise reduction failed, continuing with original audio: {e}")

            # Process user channel with Silero VAD
            print("Processing user channel with Silero VAD...")
            raw_user_vad_segments = self.detect_speech_silero_vad(audio[0], sr)

            # Process agent channel with Silero VAD
            print("Processing agent channel with Silero VAD...")
            raw_agent_vad_segments = self.detect_speech_silero_vad(audio[1], sr)
            
            # Create detailed timeline with proper overlap handling
            print("Creating detailed timeline with overlap handling...")
            
            # First normalize the segments
            user_segments = self._normalize_segments(raw_user_vad_segments, "user")
            agent_segments = self._normalize_segments(raw_agent_vad_segments, "ai_agent")
            
            # Create the detailed overlap-aware timeline
            detailed_timeline = self._create_overlap_aware_timeline(user_segments, agent_segments)
            
            # Extract clean speaker turns (non-overlaps) for traditional metrics
            user_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in detailed_timeline
                if turn["speaker"] == "user" and not turn.get("is_overlap")
            ]
            agent_speech_turns = [
                {"start": turn["start"], "end": turn["end"]}
                for turn in detailed_timeline  
                if turn["speaker"] == "ai_agent" and not turn.get("is_overlap")
            ]

            print(f"Timeline created with {len(detailed_timeline)} segments: {len(user_speech_turns)} user turns, {len(agent_speech_turns)} agent turns.")

            # Calculate VAD-based latency metrics using the detailed timeline
            vad_latency_metrics, vad_latency_details = self.calculate_latency_from_timeline(detailed_timeline)

            # Calculate overlap detection
            overlap_data = self.detect_overlaps_from_vad_segments(
                raw_user_vad_segments, raw_agent_vad_segments
            )

            # Calculate all metrics
            metrics = {
                "filename": os.path.basename(filename),
                "original_path": filename,
                "downsampled_path": output_path,
                "combined_speaker_turns": detailed_timeline,  # Use the detailed timeline
                "user_vad_segments": user_speech_turns,
                "agent_vad_segments": agent_speech_turns,
                "vad_latency_metrics": vad_latency_metrics,
                "vad_latency_details": vad_latency_details,
                "overlap_data": overlap_data,
                "ai_interrupting_user": overlap_data["has_ai_interrupting_user"],
                "user_interrupting_ai": overlap_data["has_user_interrupting_ai"],
                "talk_ratio": self.calculate_talk_ratio(user_speech_turns, agent_speech_turns),
                "average_pitch": self.calculate_average_pitch(audio, sr),
                "words_per_minute": self.calculate_words_per_minute(audio, sr, agent_speech_turns),
            }

            print(f"Successfully processed: {filename}")
            return metrics

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    def calculate_latency_from_timeline(self, detailed_timeline):
        """Calculate turn-taking latency from a pre-built detailed timeline"""
        if not detailed_timeline:
            print("No timeline segments to calculate latency.")
            return self._create_empty_latency_stats(), []

        # Calculate latencies between consecutive turns using the detailed timeline
        latencies, latency_details, overlap_stats = self._calculate_turn_latencies(detailed_timeline)

        # Generate statistics
        latency_stats = self._generate_latency_stats(latencies)
        latency_stats.update(overlap_stats)

        return latency_stats, latency_details