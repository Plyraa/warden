import numpy as np
import soundfile as sf
from pyaec import Aec
import os
import librosa
from scipy.signal import correlate

def detect_echo_segments(
    original_user_channel,
    echo_cancelled_user_channel,
    agent_channel,
    sample_rate,
    frame_duration_ms=30,
    energy_threshold=0.1,
    reduction_threshold=5.0,
    agent_energy_threshold=0.1,
    correlation_threshold=0.5,
):
    """
    Detects echo segments by analyzing energy levels and correlation.
    An echo is detected if:
    1. The agent channel has significant energy.
    2. The user channel has significant energy before cancellation.
    3. There is a significant energy reduction in the user channel after cancellation.
    4. The original user channel is correlated with the agent channel.
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(original_user_channel) // frame_size

    echo_segments = []
    in_echo_segment = False
    segment_start_frame = 0

    max_energy = np.max(original_user_channel.astype(np.float64) ** 2)
    if max_energy == 0:
        return [] # Audio is silent

    min_energy_threshold = max_energy * energy_threshold
    min_agent_energy_threshold = np.max(agent_channel.astype(np.float64) ** 2) * agent_energy_threshold

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        
        original_frame = original_user_channel[start:end].astype(np.float64)
        cancelled_frame = echo_cancelled_user_channel[start:end].astype(np.float64)
        agent_frame = agent_channel[start:end].astype(np.float64)

        original_energy = np.sum(original_frame ** 2)
        cancelled_energy = np.sum(cancelled_frame ** 2)
        agent_energy = np.sum(agent_frame ** 2)

        is_echo = False
        if agent_energy > min_agent_energy_threshold and original_energy > min_energy_threshold:
            # Calculate reduction ratio, avoid division by zero
            if cancelled_energy < 1e-6:
                reduction_ratio = original_energy
            else:
                reduction_ratio = original_energy / cancelled_energy
            
            if reduction_ratio > reduction_threshold:
                # Cross-correlation check for confirmation
                correlation = correlate(original_frame, agent_frame, mode='valid')
                if np.max(np.abs(correlation)) / (np.sqrt(original_energy * agent_energy) + 1e-9) > correlation_threshold:
                    is_echo = True

        if is_echo and not in_echo_segment:
            in_echo_segment = True
            segment_start_frame = i
        elif not is_echo and in_echo_segment:
            in_echo_segment = False
            start_time = segment_start_frame * frame_duration_ms / 1000
            end_time = i * frame_duration_ms / 1000
            echo_segments.append({"start": start_time, "end": end_time})

    if in_echo_segment:
        start_time = segment_start_frame * frame_duration_ms / 1000
        end_time = num_frames * frame_duration_ms / 1000
        echo_segments.append({"start": start_time, "end": end_time})

    return _merge_echo_segments(echo_segments)

def _merge_echo_segments(segments, max_gap_seconds=1.5):
    """Merges echo segments that are close to each other."""
    if not segments:
        return []

    # The segments are already sorted by start time
    merged = []
    current_segment = segments[0]

    for next_segment in segments[1:]:
        # If the next segment is within the max_gap, merge them
        if next_segment["start"] - current_segment["end"] <= max_gap_seconds:
            current_segment["end"] = max(current_segment["end"], next_segment["end"])
        else:
            # Otherwise, finalize the current segment and start a new one
            merged.append(current_segment)
            current_segment = next_segment
    
    # Add the last segment
    merged.append(current_segment)

    return merged

def cancel_echo_from_file(input_path: str, output_path: str):
    """
    Cancels echo from a stereo audio file.
    Left channel (user) is processed to remove echo from right channel (agent).
    """
    try:
        audio, sr = librosa.load(input_path, sr=16000, mono=False)
        if audio.shape[0] != 2:
            print(f"Audio is not stereo, skipping: {input_path}")
            return None, None, None, None

        # pyaec expects int16
        user_channel_int16 = (audio[0] * 32767).astype(np.int16)
        agent_channel_int16 = (audio[1] * 32767).astype(np.int16)

        frame_size = 160  # 10ms at 16kHz
        filter_length = 2048 # Stricter filtering
        aec = Aec(frame_size, filter_length, sr, True)

        num_frames = len(user_channel_int16) // frame_size
        cancelled_frames = []
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            rec_frame = user_channel_int16[start:end]
            echo_frame = agent_channel_int16[start:end]
            
            # Ensure frames are correct size
            if len(rec_frame) < frame_size:
                rec_frame = np.pad(rec_frame, (0, frame_size - len(rec_frame)))
            if len(echo_frame) < frame_size:
                echo_frame = np.pad(echo_frame, (0, frame_size - len(echo_frame)))

            processed_frame = aec.cancel_echo(rec_frame, echo_frame)
            cancelled_frames.append(processed_frame)

        if not cancelled_frames:
            return user_channel_int16, user_channel_int16, agent_channel_int16, None

        echo_cancelled_user_channel = np.concatenate(cancelled_frames, dtype=np.int16)
        
        # Ensure length matches original by padding if necessary
        if len(echo_cancelled_user_channel) < len(user_channel_int16):
            padding = len(user_channel_int16) - len(echo_cancelled_user_channel)
            echo_cancelled_user_channel = np.pad(echo_cancelled_user_channel, (0, padding))
        elif len(echo_cancelled_user_channel) > len(user_channel_int16):
            echo_cancelled_user_channel = echo_cancelled_user_channel[:len(user_channel_int16)]


        # Save echo-cancelled audio
        cancelled_audio_float = np.stack([
            echo_cancelled_user_channel.astype(np.float32) / 32767.0,
            audio[1] # original agent channel
        ])
        sf.write(output_path, cancelled_audio_float.T, sr)

        return user_channel_int16, echo_cancelled_user_channel, agent_channel_int16, sr

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None, None, None, None

def process_audio_for_echo(input_path, output_dir, params=None):
    if params is None:
        params = {}
        
    filename = os.path.basename(input_path)
    output_filename = os.path.splitext(filename)[0] + "_echo_cancelled.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    original_user, cancelled_user, agent_channel, sr = cancel_echo_from_file(input_path, output_path)

    if original_user is None:
        return None

    echo_segments = detect_echo_segments(original_user, cancelled_user, agent_channel, sr, **params)
    
    return {
        "filename": filename,
        "echo_cancelled_path": output_path,
        "echo_segments": echo_segments,
    }
