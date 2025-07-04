"""
Facebook Denoiser implementation for noise reduction on user audio channel
"""
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import os
import argparse

class FacebookDenoiser:
    def __init__(self):
        self.denoiser_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000  # Facebook denoiser expects 16kHz
        print(f"Initializing Facebook Denoiser on {self.device}")
    
    def load_model(self) -> None:
        """Load the Facebook denoiser model"""
        try:
            from denoiser.pretrained import add_model_flags, get_model
            
            print("Loading Facebook Denoiser model (this may take a while on first run)...")
            
            # Create argument parser and add model flags
            parser = argparse.ArgumentParser()
            add_model_flags(parser)
            
            # Parse empty args to get defaults, then set model to dns64
            args = parser.parse_args([])
            args.dns64 = True  # Use the DNS64 model
            
            # Get the pretrained model
            self.denoiser_model = get_model(args).to(self.device)
            self.denoiser_model.eval()  # Set to evaluation mode
            print("Facebook Denoiser model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Facebook Denoiser model: {e}")
    
    def get_model(self):
        """Get the denoiser model, loading it if necessary"""
        if self.denoiser_model is None:
            self.load_model()
        return self.denoiser_model
    
    def denoise_audio_channel(self, audio_channel: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to a single audio channel using Facebook's denoiser
        
        Args:
            audio_channel: 1D numpy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Denoised audio channel as numpy array
        """
        try:
            model = self.get_model()
            
            # Convert to torch tensor
            audio_tensor = torch.FloatTensor(audio_channel)
            
            # Ensure audio is on the correct device
            audio_tensor = audio_tensor.to(self.device)
            
            # Resample if necessary (denoiser expects 16kHz)
            if sample_rate != self.sample_rate:
                print(f"Resampling audio from {sample_rate}Hz to {self.sample_rate}Hz for denoising")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.sample_rate
                ).to(self.device)
                audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
                original_sr = sample_rate
                sample_rate = self.sample_rate
            else:
                original_sr = sample_rate
            
            # Ensure correct shape for the model (batch_size, channels, length)
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim
            
            print("Applying Facebook's Denoiser to user channel...")
            
            # Apply denoising
            with torch.no_grad():
                denoised_tensor = model(audio_tensor)
            
            # Remove batch and channel dimensions and convert back to numpy
            denoised_audio = denoised_tensor.squeeze().cpu().numpy()
            
            # Resample back to original sample rate if needed
            if original_sr != self.sample_rate:
                print(f"Resampling denoised audio back to {original_sr}Hz")
                resampler_back = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=original_sr
                ).to(self.device)
                denoised_tensor_back = resampler_back(
                    torch.FloatTensor(denoised_audio).unsqueeze(0).to(self.device)
                ).squeeze(0)
                denoised_audio = denoised_tensor_back.cpu().numpy()
            
            print("Denoising completed successfully")
            return denoised_audio
            
        except Exception as e:
            print(f"Error during denoising: {e}")
            print("Falling back to original audio without denoising")
            return audio_channel
    
    def apply_noise_reduction_to_stereo(self, stereo_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction only to the left channel (user channel) of stereo audio
        
        Args:
            stereo_audio: 2D numpy array with shape (2, samples) - [left_channel, right_channel]
            sample_rate: Sample rate of the audio
            
        Returns:
            Stereo audio with denoised left channel and original right channel
        """
        if len(stereo_audio.shape) != 2 or stereo_audio.shape[0] != 2:
            raise ValueError(f"Expected stereo audio with shape (2, samples), got {stereo_audio.shape}")
        
        print("Applying noise reduction to left channel (user channel) only...")
        
        # Extract channels
        left_channel = stereo_audio[0]  # User channel
        right_channel = stereo_audio[1]  # Agent channel - keep original
        
        # Apply denoising only to left channel (user)
        denoised_left = self.denoise_audio_channel(left_channel, sample_rate)
        
        # Ensure same length (in case of slight differences due to resampling)
        min_length = min(len(denoised_left), len(right_channel))
        denoised_left = denoised_left[:min_length]
        right_channel = right_channel[:min_length]
        
        # Reconstruct stereo audio with denoised left channel
        denoised_stereo = np.array([denoised_left, right_channel])
        
        print(f"Noise reduction applied. Output shape: {denoised_stereo.shape}")
        return denoised_stereo


# Global instance for reuse across multiple audio files
_denoiser_instance = None

def get_denoiser_instance() -> FacebookDenoiser:
    """Get a singleton instance of the Facebook denoiser"""
    global _denoiser_instance
    if _denoiser_instance is None:
        _denoiser_instance = FacebookDenoiser()
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
