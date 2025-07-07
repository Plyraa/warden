"""
DeepFilterNet implementation for noise reduction on user audio channel
Lightweight neural network alternative optimized for speech enhancement
"""
import numpy as np
import librosa
import torch
import tempfile
import os
from typing import Tuple, Optional

# Try to import DeepFilterNet
try:
    from df.enhance import enhance, init_df
    from df.utils import download_file
    import soundfile as sf
    DEEPFILTERNET_AVAILABLE = True
    print("✅ DeepFilterNet available - Using efficient neural denoiser")
except ImportError:
    DEEPFILTERNET_AVAILABLE = False
    print("❌ DeepFilterNet not available. Install with: pip install deepfilternet")
    print("Falling back to lightweight spectral processing")

class DeepFilterNetDenoiser:
    """DeepFilterNet - Modern lightweight neural network for noise suppression"""
    
    def __init__(self, model_base_dir=None):
        if not DEEPFILTERNET_AVAILABLE:
            raise ImportError("DeepFilterNet not available. Install with: pip install deepfilternet")
        
        self.model = None
        self.df_state = None
        self.sr = 48000  # DeepFilterNet works at 48kHz
        self.model_base_dir = model_base_dir
        print("Initialized DeepFilterNet denoiser (RAM usage: 200-400MB)")
        
    def load_model(self):
        """Load DeepFilterNet model"""
        try:
            print("Loading DeepFilterNet model...")
            self.model, self.df_state, _ = init_df(
                model_base_dir=self.model_base_dir,
                post_filter=True,  # Enable post-filter for better quality
                log_level="WARNING"
            )
            print("DeepFilterNet model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepFilterNet: {e}")
    
    def denoise_audio_channel(self, audio_channel: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply DeepFilterNet to a single audio channel
        
        Args:
            audio_channel: 1D numpy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Denoised audio channel as numpy array
        """
        try:
            if self.model is None:
                self.load_model()
            
            # Resample to 48kHz if needed (DeepFilterNet requirement)
            if sample_rate != self.sr:
                print(f"Resampling audio from {sample_rate}Hz to {self.sr}Hz for DeepFilterNet")
                audio_48k = librosa.resample(
                    audio_channel,
                    orig_sr=sample_rate,
                    target_sr=self.sr
                )
            else:
                audio_48k = audio_channel.copy()
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_48k).float().unsqueeze(0)  # Add batch dim
            
            # Apply enhancement
            print("Applying DeepFilterNet enhancement...")
            with torch.no_grad():
                enhanced_tensor = enhance(
                    self.model, 
                    self.df_state, 
                    audio_tensor
                )
            
            # Convert back to numpy
            enhanced_audio = enhanced_tensor.squeeze(0).numpy()
            
            # Resample back to original sample rate if needed
            if sample_rate != self.sr:
                print(f"Resampling enhanced audio back to {sample_rate}Hz")
                enhanced_audio = librosa.resample(
                    enhanced_audio,
                    orig_sr=self.sr,
                    target_sr=sample_rate
                )
            
            # Trim to original length to handle any slight differences
            return enhanced_audio[:len(audio_channel)]
            
        except Exception as e:
            print(f"DeepFilterNet processing failed: {e}")
            return audio_channel
    
    def apply_noise_reduction_to_stereo(self, stereo_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply DeepFilterNet only to the left channel (user channel) of stereo audio
        
        Args:
            stereo_audio: 2D numpy array with shape (2, samples) - [left_channel, right_channel]
            sample_rate: Sample rate of the audio
            
        Returns:
            Stereo audio with denoised left channel and original right channel
        """
        if len(stereo_audio.shape) != 2 or stereo_audio.shape[0] != 2:
            raise ValueError(f"Expected stereo audio with shape (2, samples), got {stereo_audio.shape}")
        
        print("Applying DeepFilterNet to left channel (user channel) only...")
        
        # Extract channels
        left_channel = stereo_audio[0]  # User channel
        right_channel = stereo_audio[1]  # Agent channel - keep original
        
        # Apply DeepFilterNet only to left channel
        denoised_left = self.denoise_audio_channel(left_channel, sample_rate)
        
        # Ensure same length
        min_length = min(len(denoised_left), len(right_channel))
        denoised_left = denoised_left[:min_length]
        right_channel = right_channel[:min_length]
        
        # Reconstruct stereo audio with denoised left channel
        denoised_stereo = np.array([denoised_left, right_channel])
        
        print(f"DeepFilterNet processing completed. Output shape: {denoised_stereo.shape}")
        return denoised_stereo


class LightweightSpectralDenoiser:
    """Fallback implementation when RNNoise is not available"""
    
    def __init__(self):
        print("Initialized lightweight spectral denoiser (RAM usage: <10MB)")
        
    def denoise_audio_channel(self, audio_channel: np.ndarray, sample_rate: int) -> np.ndarray:
        """Lightweight spectral subtraction"""
        try:
            # Simple bandpass filter for speech (300-3400 Hz)
            from scipy.signal import butter, filtfilt
            
            nyquist = sample_rate / 2
            low, high = 300/nyquist, min(3400/nyquist, 0.99)
            
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, audio_channel)
            
            # Simple noise gate
            threshold = np.percentile(np.abs(filtered), 20)  # Bottom 20% as noise floor
            gate_mask = np.abs(filtered) > threshold
            
            # Smooth the gate to avoid clicks
            from scipy.ndimage import gaussian_filter1d
            gate_smooth = gaussian_filter1d(gate_mask.astype(float), sigma=100)
            
            # Apply gate with some noise floor preservation
            enhanced = filtered * (gate_smooth * 0.95 + 0.05)
            
            return enhanced
            
        except Exception as e:
            print(f"Lightweight processing failed: {e}")
            return audio_channel
    
    def apply_noise_reduction_to_stereo(self, stereo_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply lightweight processing only to left channel"""
        if len(stereo_audio.shape) != 2 or stereo_audio.shape[0] != 2:
            raise ValueError(f"Expected stereo audio with shape (2, samples), got {stereo_audio.shape}")
        
        print("Applying lightweight noise reduction to left channel (user channel) only...")
        
        left_channel = stereo_audio[0]
        right_channel = stereo_audio[1]
        
        denoised_left = self.denoise_audio_channel(left_channel, sample_rate)
        
        min_length = min(len(denoised_left), len(right_channel))
        denoised_left = denoised_left[:min_length]
        right_channel = right_channel[:min_length]
        
        denoised_stereo = np.array([denoised_left, right_channel])
        
        print(f"Lightweight noise reduction completed. Output shape: {denoised_stereo.shape}")
        return denoised_stereo


# Smart denoiser that uses DeepFilterNet if available, falls back to lightweight processing
class SmartDenoiser:
    """Smart denoiser that automatically chooses the best available method"""
    
    def __init__(self):
        self.denoiser = None
        self.method_name = "None"
        
        # Try to use DeepFilterNet first
        if DEEPFILTERNET_AVAILABLE:
            try:
                self.denoiser = DeepFilterNetDenoiser()
                self.method_name = "DeepFilterNet"
            except Exception as e:
                print(f"Failed to initialize DeepFilterNet: {e}")
                self.denoiser = None
        
        # Fallback to lightweight spectral processing
        if self.denoiser is None:
            self.denoiser = LightweightSpectralDenoiser()
            self.method_name = "Lightweight Spectral"
        
        print(f"🎯 Active noise reduction method: {self.method_name}")
    
    def denoise_audio_channel(self, audio_channel: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to a single audio channel"""
        return self.denoiser.denoise_audio_channel(audio_channel, sample_rate)
    
    def apply_noise_reduction_to_stereo(self, stereo_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction only to the left channel (user channel) of stereo audio"""
        return self.denoiser.apply_noise_reduction_to_stereo(stereo_audio, sample_rate)


# Global instance for reuse across multiple audio files
_denoiser_instance = None

def get_denoiser_instance() -> SmartDenoiser:
    """Get a singleton instance of the smart denoiser"""
    global _denoiser_instance
    if _denoiser_instance is None:
        _denoiser_instance = SmartDenoiser()
    return _denoiser_instance

def apply_noise_reduction(stereo_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Convenience function to apply noise reduction to stereo audio
    
    Args:
        stereo_audio: 2D numpy array with shape (2, samples) - [left_channel, right_channel]
        sample_rate: Sample rate of the audio
        
    Returns:
        Stereo audio with denoised left channel and original right channel
    """
    denoiser = get_denoiser_instance()
    return denoiser.apply_noise_reduction_to_stereo(stereo_audio, sample_rate)

# Backward compatibility alias
FacebookDenoiser = SmartDenoiser
