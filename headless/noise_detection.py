import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import find_peaks

# --- Constants ---
SAMPLE_RATE = 16000
NOISE_TEST_DIR = "noise_test"

# SNR calculation constants
SNR_THRESHOLD = 35  # SNR value in dB below which the audio is considered noisy, you may try 25-50 dB

# VAD constants
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_ENERGY_THRESHOLD = 0.01  # Energy threshold for speech detection

# Noise interrupt constants
NOISE_PEAK_THRESHOLD = 0.05 # Amplitude threshold for detecting noise peaks
INTERRUPTION_WINDOW_MS = 500 # Window to check for agent silence after a noise peak
INTERRUPTION_WINDOW_SAMPLES = int(SAMPLE_RATE * INTERRUPTION_WINDOW_MS / 1000)


def is_speaking(frame):
    """A simple energy-based Voice Activity Detection (VAD)."""
    rms_energy = np.sqrt(np.mean(np.square(frame)))
    return rms_energy > VAD_ENERGY_THRESHOLD

def calculate_snr(audio_channel):
    """Calculates the Signal-to-Noise Ratio (SNR) of an audio channel."""
    # Separate speech and noise using a simple energy-based VAD
    speech_frames = []
    noise_frames = []
    for i in range(0, len(audio_channel) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES):
        frame = audio_channel[i:i+VAD_FRAME_SAMPLES]
        if is_speaking(frame):
            speech_frames.append(frame)
        else:
            noise_frames.append(frame)

    if not speech_frames or not noise_frames:
        return float('inf') # Not enough data to calculate SNR

    speech_power = np.mean(np.square(np.concatenate(speech_frames)))
    noise_power = np.mean(np.square(np.concatenate(noise_frames)))

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(speech_power / noise_power)
    return snr

def detect_noise_interrupt(user_channel, agent_channel):
    """
    Detects if a noise peak from the user channel likely caused the agent to stop speaking.
    """
    # 1. Get VAD for the entire agent channel
    agent_vad = [is_speaking(agent_channel[i:i+VAD_FRAME_SAMPLES])
                 for i in range(0, len(agent_channel) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES)]

    # 2. Find where the agent stops speaking
    speech_stop_indices = [i for i, (prev, curr) in enumerate(zip(agent_vad, agent_vad[1:])) if prev and not curr]

    if not speech_stop_indices:
        return False

    # 3. Find noise peaks in the user's channel
    noise_peaks, _ = find_peaks(np.abs(user_channel), height=NOISE_PEAK_THRESHOLD, distance=SAMPLE_RATE)

    # 4. Check if any speech stop happens right after a noise peak
    for stop_frame_index in speech_stop_indices:
        stop_sample_index = stop_frame_index * VAD_FRAME_SAMPLES
        for peak_sample_index in noise_peaks:
            if 0 < (stop_sample_index - peak_sample_index) < INTERRUPTION_WINDOW_SAMPLES:
                return True # Found a potential noise interrupt

    return False

def detect_noise(audio_path):
    """
    Detects noise and noise-based interruptions in a stereo audio file.
    """
    result = {
        "file": os.path.basename(audio_path),
        "hasNoise": False,
        "noiseInterrupt": False,
        "error": None
    }

    try:
        audio, sr = sf.read(audio_path, dtype="float32")

        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=SAMPLE_RATE).T

        if audio.ndim != 2 or audio.shape[1] != 2:
            result["error"] = "Audio is not stereo."
            return result

        user_channel, agent_channel = audio[:, 0], audio[:, 1]

        # Calculate SNR to determine if there is noise
        snr = calculate_snr(user_channel)
        if snr < SNR_THRESHOLD:
            result["hasNoise"] = True

        # Detect noise interruption
        if result["hasNoise"]:
            result["noiseInterrupt"] = detect_noise_interrupt(user_channel, agent_channel)

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        result["error"] = str(e)

    return result

if __name__ == "__main__":
    if not os.path.exists(NOISE_TEST_DIR):
        print(f"Error: Directory '{NOISE_TEST_DIR}' not found.")
    else:
        print("-" * 80)
        print(f"Analyzing audio files in '{NOISE_TEST_DIR}' for noise...")
        print("-" * 80)

        results = []
        for filename in sorted(os.listdir(NOISE_TEST_DIR)):
            if filename.lower().endswith(('.wav', '.mp3')):
                file_path = os.path.join(NOISE_TEST_DIR, filename)
                res = detect_noise(file_path)
                results.append(res)

        print(f"{'File Name':<40} {'Has Noise':<12} {'Noise Interrupt':<16} {'Error'}")
        print("=" * 80)
        for res in results:
            status = "Yes" if res.get("hasNoise") else "No"
            interrupt_status = "Yes" if res.get("noiseInterrupt") else "No"
            details = res.get("error", "N/A")
            print(f"{res['file']:<40} {status:<12} {interrupt_status:<16} {details}")
        print("-" * 80)
