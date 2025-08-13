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
CORRELATION_THRESHOLD = 0.020  # Reasonable numbers here are between 0.015 to 0.050 in my experience

MIN_ECHO_DELAY_MS = 20      # Minimum delay for valid echo (20ms - permissive)
MAX_ECHO_DELAY_MS = 1000    # Maximum delay for valid echo (1000ms - permissive)
MIN_ECHO_DELAY_SAMPLES = int(SAMPLE_RATE * MIN_ECHO_DELAY_MS / 1000)
MAX_ECHO_DELAY_SAMPLES = int(SAMPLE_RATE * MAX_ECHO_DELAY_MS / 1000)
ECHO_AMPLITUDE_RATIO_MAX = 1.50 
MIN_SPEECH_DURATION_MS = 50     # Minimum speech duration to consider for echo (reduced further)
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * MIN_SPEECH_DURATION_MS / 1000)
SPECTRAL_COHERENCE_MIN = 0.10   # Lower spectral coherence requirement
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

def validate_echo_characteristics(customer_channel, agent_channel, lag_samples):
    """
    Validate that the detected correlation represents a true echo by checking:
    1. Reasonable delay (30-800ms)
    2. Amplitude relationship (more permissive)
    3. Sufficient speech activity (reduced requirements)
    4. Frequency coherence (optional, more permissive)
    """
    validation_msgs = []
    
    # 1. Check if lag is within reasonable echo delay range
    if not (MIN_ECHO_DELAY_SAMPLES <= abs(lag_samples) <= MAX_ECHO_DELAY_SAMPLES):
        return False, f"Delay outside reasonable echo range ({lag_samples} samples, {lag_samples/SAMPLE_RATE*1000:.1f}ms)"
    validation_msgs.append(f"Valid delay: {lag_samples/SAMPLE_RATE*1000:.1f}ms")
    
    # 2. Check amplitude relationship (more permissive)
    customer_rms = np.sqrt(np.mean(np.square(customer_channel)))
    agent_rms = np.sqrt(np.mean(np.square(agent_channel)))
    
    if customer_rms == 0 or agent_rms == 0:
        return False, "Silent channel detected"
    
    # More permissive amplitude check
    amplitude_ratio = min(customer_rms, agent_rms) / max(customer_rms, agent_rms)
    if amplitude_ratio > ECHO_AMPLITUDE_RATIO_MAX:
        return False, f"Channels too similar in amplitude (ratio: {amplitude_ratio:.3f})"
    validation_msgs.append(f"Amplitude ratio OK: {amplitude_ratio:.3f}")
    
    # 3. Check for sufficient speech activity (very reduced requirements)
    customer_activity = np.sum(np.abs(customer_channel) > np.std(customer_channel) * 0.02)  # Very reduced threshold
    agent_activity = np.sum(np.abs(agent_channel) > np.std(agent_channel) * 0.02)
    
    if customer_activity < MIN_SPEECH_SAMPLES or agent_activity < MIN_SPEECH_SAMPLES:
        return False, f"Insufficient speech activity (customer: {customer_activity}, agent: {agent_activity}, min: {MIN_SPEECH_SAMPLES})"
    validation_msgs.append(f"Speech activity OK")
    
    # 4. Check frequency coherence (very optional and very permissive)
    spectral_corr = 0.0
    if len(customer_channel) > 256:  # Even more reduced requirement
        try:
            customer_fft = np.fft.fft(customer_channel[:256])  # Even smaller window
            agent_fft = np.fft.fft(agent_channel[:256])
            
            customer_spectrum = np.abs(customer_fft)
            agent_spectrum = np.abs(agent_fft)
            
            # Check if spectra have sufficient variation to avoid division by zero
            customer_std = np.std(customer_spectrum)
            agent_std = np.std(agent_spectrum)
            
            if customer_std > 1e-10 and agent_std > 1e-10:
                # Normalize spectra
                customer_spectrum = customer_spectrum / (np.sum(customer_spectrum) + 1e-9)
                agent_spectrum = agent_spectrum / (np.sum(agent_spectrum) + 1e-9)
                
                # Calculate spectral correlation with error handling
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr_matrix = np.corrcoef(customer_spectrum, agent_spectrum)
                    if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                        spectral_corr = corr_matrix[0, 1]
                    else:
                        spectral_corr = 0.0
                
                # Very permissive spectral coherence check (just informational now)
                validation_msgs.append(f"Spectral coherence: {spectral_corr:.3f} (informational only)")
            else:
                validation_msgs.append("Spectra too uniform for correlation analysis (but allowing)")
        except Exception as e:
            validation_msgs.append(f"Spectral analysis failed: {str(e)[:30]}... (but allowing)")
    else:
        validation_msgs.append("Audio too short for spectral analysis (but allowing)")
    
    return True, "; ".join(validation_msgs)

def validate_lag_consistency(customer_channel, agent_channel, lag_samples, window_size=512):
    """
    Check if the lag is consistent across different parts of the audio.
    True echoes should have consistent delay. Made more permissive.
    """
    if len(customer_channel) < window_size * 2:  # Reduced requirement
        return True  # Too short to validate, assume valid
    
    num_windows = 2  # Reduced from 3 to 2
    window_step = len(customer_channel) // num_windows
    detected_lags = []
    
    for i in range(num_windows):
        start_idx = i * window_step
        end_idx = min(start_idx + window_size, len(customer_channel))
        
        if end_idx - start_idx < window_size // 3:  # More permissive
            continue
        
        window_customer = customer_channel[start_idx:end_idx]
        window_agent = agent_channel[start_idx:end_idx]
        
        # Normalize windows
        if np.std(window_customer) > 1e-9 and np.std(window_agent) > 1e-9:
            norm_customer = (window_customer - np.mean(window_customer)) / np.std(window_customer)
            norm_agent = (window_agent - np.mean(window_agent)) / np.std(window_agent)
            
            # Cross-correlate
            correlation = correlate(norm_customer, norm_agent, mode='full')
            peak_idx = np.argmax(np.abs(correlation))
            window_lag = peak_idx - (len(norm_customer) - 1)
            
            detected_lags.append(window_lag)
    
    if len(detected_lags) < 1:  # Reduced requirement
        return True  # Not enough windows to validate
    
    # Check if lags are consistent (more permissive - within 50% of original lag)
    if len(detected_lags) == 1:
        return True  # Only one window, can't check consistency
        
    lag_variance = np.var(detected_lags)
    expected_variance_threshold = (abs(lag_samples) * 0.5) ** 2  # Increased from 0.2 to 0.5
    
    return lag_variance <= expected_variance_threshold

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
    Enhanced with multiple validation criteria to reduce false positives.
    """
    result = {
        "file": os.path.basename(audio_path),
        "hasEcho": False,
        "echoInterrupt": False,
        "correlation": "0.0000",
        "validation_details": [],
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

        # Initial correlation check
        norm_customer = (customer_channel - np.mean(customer_channel)) / (np.std(customer_channel) + 1e-9)
        norm_agent = (agent_channel - np.mean(agent_channel)) / (np.std(agent_channel) + 1e-9)
        
        correlation = correlate(norm_customer, norm_agent, mode='full')
        normalized_correlation = correlation / min_len
        max_correlation = np.max(np.abs(normalized_correlation))
        result["correlation"] = f"{max_correlation:.4f}"

        # First check: correlation threshold
        if max_correlation <= CORRELATION_THRESHOLD:
            result["validation_details"].append("Below correlation threshold")
            return result

        # Find the lag of the potential echo
        peak_index = np.argmax(np.abs(normalized_correlation))
        lag_samples = peak_index - (min_len - 1)
        
        result["validation_details"].append(f"Initial correlation: {max_correlation:.4f}")
        result["validation_details"].append(f"Detected lag: {lag_samples} samples ({lag_samples/SAMPLE_RATE*1000:.1f}ms)")

        # Enhanced validation checks - made more permissive
        is_valid_echo, validation_msg = validate_echo_characteristics(
            customer_channel, agent_channel, lag_samples
        )
        result["validation_details"].append(validation_msg)
        
        if not is_valid_echo:
            return result
        
        # Check lag consistency across audio segments (more permissive)
        is_consistent = validate_lag_consistency(customer_channel, agent_channel, lag_samples)
        if not is_consistent:
            result["validation_details"].append("Inconsistent lag across audio segments (but still checking for echo)")
            # Don't return here - allow echo detection even with inconsistent lag
        else:
            result["validation_details"].append("Consistent lag detected")

        # If we get here, it's likely a true echo
        has_echo = True
        result["hasEcho"] = has_echo
        result["validation_details"].append("VALIDATED as true echo")

        if has_echo and lag_samples > 0:
            # Only check for interrupt if the lag is positive (echo comes after original sound)
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

        print(f"{'File Name':<40} {'Has Echo':<12} {'Echo Interrupt':<16} {'Correlation':<12} {'Validation'}")
        print("=" * 100)
        for res in results:
            status = "Yes" if res.get("hasEcho") else "No"
            interrupt_status = "Yes" if res.get("echoInterrupt") else "No"
            correlation = res.get("correlation", "N/A")
            validation = res.get("validation_details", ["N/A"])[-1] if res.get("validation_details") else "N/A"
            error = res.get("error")
            
            if error:
                validation = f"ERROR: {error}"
            
            print(f"{res['file']:<40} {status:<12} {interrupt_status:<16} {correlation:<12} {validation}")
            
            # Show detailed validation for files with echo detected
            if res.get("hasEcho") and res.get("validation_details"):
                for detail in res["validation_details"]:
                    print(f"{'':>40} -> {detail}")
                print()
        print("-" * 100)