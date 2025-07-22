import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import correlate, find_peaks

# --- Constants ---
# The sample rate to use for all audio processing
SAMPLE_RATE = 16000
# Directory containing the audio files to be analyzed
ECHO_EXAMPLES_DIR = "echo_examples"
# The correlation value above which a file is considered to have echo.
CORRELATION_THRESHOLD = 0.025
# VAD constants
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_ENERGY_THRESHOLD = 0.01  # Energy threshold for speech detection
VAD_SILENCE_WINDOW_MS = 200  # Window to check for silence after an echo
VAD_SILENCE_SAMPLES = int(SAMPLE_RATE * VAD_SILENCE_WINDOW_MS / 1000)

def is_speaking(frame):
    """A simple energy-based Voice Activity Detection (VAD)."""
    rms_energy = np.sqrt(np.mean(np.square(frame)))
    return rms_energy > VAD_ENERGY_THRESHOLD

def detect_echo_interrupt(agent_channel, lag_samples):
    """
    Detects if an echo likely caused the agent to stop speaking.
    """
    # 1. Get VAD for the entire agent channel
    vad_frames = [is_speaking(agent_channel[i:i+VAD_FRAME_SAMPLES]) 
                  for i in range(0, len(agent_channel) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES)]

    # 2. Find where the agent stops speaking (transition from True to False)
    speech_stop_indices = [i for i, (prev, curr) in enumerate(zip(vad_frames, vad_frames[1:])) if prev and not curr]

    if not speech_stop_indices:
        return False

    # 3. Find peaks in the agent's speech (potential echo sources)
    agent_speech_peaks, _ = find_peaks(agent_channel, height=0.1, distance=SAMPLE_RATE)

    # 4. Check if any speech stop happens right after an echo returns
    for stop_frame_index in speech_stop_indices:
        stop_sample_index = stop_frame_index * VAD_FRAME_SAMPLES
        
        for peak_sample_index in agent_speech_peaks:
            # Time the echo of this peak returns
            echo_return_time = peak_sample_index + lag_samples
            
            # Check if the agent stopped speaking within a certain window AFTER the echo
            if 0 < (stop_sample_index - echo_return_time) < VAD_SILENCE_SAMPLES:
                return True # Found a potential echo interrupt
                
    return False

def detect_echo(audio_path, apply_noise_reduction=False):
    """
    Detects echo and echo-based interruptions in a stereo audio file.
    """
    result = {
        "file": os.path.basename(audio_path),
        "hasEcho": False,
        "echoInterrupt": False,
        "correlation": "0.0000",
        "error": None
    }

    try:
        audio, sr = sf.read(audio_path, dtype="float32")

        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=SAMPLE_RATE).T
        
        if audio.ndim != 2 or audio.shape[1] != 2:
            result["error"] = "Audio is not stereo."
            return result

        customer_channel, agent_channel = audio[:, 0], audio[:, 1]

        if apply_noise_reduction:
            from noise_reduction import apply_noise_reduction as denoise
            temp_stereo = np.array([customer_channel, agent_channel])
            denoised_stereo = denoise(temp_stereo, SAMPLE_RATE)
            customer_channel = denoised_stereo[0]

        min_len = min(len(customer_channel), len(agent_channel))
        customer_channel = customer_channel[:min_len]
        agent_channel = agent_channel[:min_len]

        norm_customer = (customer_channel - np.mean(customer_channel)) / (np.std(customer_channel) + 1e-9)
        norm_agent = (agent_channel - np.mean(agent_channel)) / (np.std(agent_channel) + 1e-9)
        
        correlation = correlate(norm_customer, norm_agent, mode='full')
        normalized_correlation = correlation / min_len
        max_correlation = np.max(np.abs(normalized_correlation))
        result["correlation"] = f"{max_correlation:.4f}"

        has_echo = max_correlation > CORRELATION_THRESHOLD
        result["hasEcho"] = has_echo

        if has_echo:
            # Find the lag of the echo
            peak_index = np.argmax(np.abs(normalized_correlation))
            lag_samples = peak_index - (min_len - 1)
            
            # Only check for interrupt if the lag is positive (echo comes after original sound)
            if lag_samples > 0:
                result["echoInterrupt"] = detect_echo_interrupt(agent_channel, lag_samples)

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        result["error"] = str(e)

    return result

if __name__ == "__main__":
    if not os.path.exists(ECHO_EXAMPLES_DIR):
        print(f"Error: Directory '{ECHO_EXAMPLES_DIR}' not found.")
    else:
        print("-" * 80)
        print(f"Analyzing audio files in '{ECHO_EXAMPLES_DIR}' for echo...")
        print("-" * 80)
        
        results = []
        for filename in sorted(os.listdir(ECHO_EXAMPLES_DIR)):
            if filename.lower().endswith(('.wav', '.mp3')):
                file_path = os.path.join(ECHO_EXAMPLES_DIR, filename)
                res = detect_echo(file_path, apply_noise_reduction=True)
                results.append(res)

        print(f"{'File Name':<40} {'Has Echo':<12} {'Echo Interrupt':<16} {'Correlation/Error'}")
        print("=" * 80)
        for res in results:
            status = "Yes" if res.get("hasEcho") else "No"
            interrupt_status = "Yes" if res.get("echoInterrupt") else "No"
            details = res.get("correlation") or res.get("error", "N/A")
            print(f"{res['file']:<40} {status:<12} {interrupt_status:<16} {details}")
        print("-" * 80)