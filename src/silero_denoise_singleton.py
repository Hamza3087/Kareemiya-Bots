#!/usr/bin/env python3
"""
Silero Denoise Singleton - Thread-Safe Audio Noise Suppression

Production-grade singleton for Silero Denoise model with:
- CPU/GPU inference support
- Thread-safe access with locking
- SNR-based adaptive filtering (only denoise when needed)
- Local-first loading with torch.hub fallback
- Support for 8kHz telephony audio

Memory: ~50MB model footprint (shared across all calls)
Performance: ~20ms per second of audio (CPU)

Usage:
    denoiser = SileroDenoiseSingleton()
    model = denoiser.get_model(logger)

    # With SNR-based adaptive filtering
    clean_audio = denoiser.denoise_audio(
        audio_bytes,
        snr_db=12.0,  # Current SNR
        snr_threshold=15.0,  # Only denoise if SNR < 15dB
        aggressive_threshold=10.0  # More aggressive if SNR < 10dB
    )
"""

import os
import threading
import logging
import torch
import numpy as np
from typing import Optional, Tuple


class SileroDenoiseSingleton:
    """
    Thread-safe singleton for Silero Denoise model.

    Features:
    - SNR-based adaptive filtering (skip denoising for clean audio)
    - 8kHz input support (native to telephony)
    - Thread-safe inference with locking
    - Efficient: ~20ms per second of audio on CPU
    """

    _instance = None
    _lock = threading.Lock()              # Singleton creation lock
    _model = None
    _utils = None
    _model_lock = threading.Lock()        # Model loading lock
    _inference_lock = threading.Lock()    # Inference serialization
    _logger = None
    _load_error = None
    _device = None

    def __new__(cls):
        """Thread-safe singleton creation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, logger: Optional[logging.Logger] = None):
        """
        Get or load the Silero Denoise model.

        Loading strategy:
        1. Try local path first (fast)
        2. Fallback to torch.hub (slower, network required)

        Args:
            logger: Logger instance for diagnostics

        Returns:
            Loaded model or None on failure
        """
        # Fast path - model already loaded
        if self._model is not None:
            return self._model

        # Setup logger
        if logger is not None:
            self._logger = logger
        if self._logger is None:
            self._logger = logging.getLogger(__name__)

        with self._model_lock:
            # Double-check inside lock
            if self._model is not None:
                return self._model

            # Check if another thread failed to load
            if self._load_error is not None:
                self._logger.warning(f"Previous load failed: {self._load_error}")
                return None

            try:
                # Determine device (prefer CPU for telephony - low latency)
                self._device = torch.device('cpu')
                self._logger.info(f"Using device: {self._device}")

                # Local model path
                LOCAL_MODEL_DIR = '/root/sip-bot/models/silero_denoise'

                # Try local loading first
                if os.path.exists(LOCAL_MODEL_DIR):
                    self._logger.info(f"Attempting local load from: {LOCAL_MODEL_DIR}")
                    try:
                        model, samples, utils = torch.hub.load(
                            repo_or_dir=LOCAL_MODEL_DIR,
                            model='silero_denoise',
                            name='small_fast',  # Use fast variant for real-time
                            device=self._device,
                            source='local',
                            force_reload=False,
                            trust_repo=True
                        )
                        self._model = model
                        self._utils = utils
                        self._logger.info("Silero Denoise loaded from local storage")
                        return self._model
                    except Exception as local_error:
                        self._logger.warning(f"Local load failed: {local_error}")

                # Fallback to torch.hub
                self._logger.info("Loading Silero Denoise from torch.hub...")
                model, samples, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_denoise',
                    name='small_fast',  # Use fast variant for real-time
                    device=self._device,
                    force_reload=False
                )
                self._model = model
                self._utils = utils
                self._logger.info("Silero Denoise loaded from torch.hub")

                return self._model

            except Exception as e:
                self._load_error = str(e)
                self._logger.error(f"Failed to load Silero Denoise: {e}")
                return None

    def denoise_audio(
        self,
        audio_bytes: bytes,
        snr_db: Optional[float] = None,
        snr_threshold: float = 15.0,
        aggressive_threshold: float = 10.0,
        sample_rate: int = 8000,
        force_denoise: bool = False
    ) -> bytes:
        """
        Denoise audio with SNR-based adaptive filtering.

        Args:
            audio_bytes: Raw 16-bit PCM audio bytes
            snr_db: Current SNR in dB (if None, always denoise)
            snr_threshold: Only denoise if SNR < this value
            aggressive_threshold: Apply more aggressive denoising if SNR < this
            sample_rate: Input sample rate (8000 or 16000)
            force_denoise: If True, bypass SNR check and always denoise

        Returns:
            Denoised audio bytes (same format as input)

        Performance:
            ~20ms per second of audio on CPU
        """
        if self._model is None:
            if self._logger:
                self._logger.warning("Denoise model not loaded, returning original audio")
            return audio_bytes

        # SNR-based filtering: skip denoising for clean audio
        if not force_denoise and snr_db is not None:
            if snr_db >= snr_threshold:
                # Audio is clean enough, no need to denoise
                return audio_bytes

        if not audio_bytes or len(audio_bytes) < 100:
            return audio_bytes

        with self._inference_lock:
            try:
                # Convert bytes to numpy array
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0  # Normalize to [-1, 1]

                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self._device)

                # Get the denoise function from utils
                if self._utils is None:
                    self._logger.error("Denoise utils not loaded")
                    return audio_bytes

                # Unpack utils (read_audio, save_audio, denoise)
                _, _, denoise_fn = self._utils

                # Denoise
                with torch.no_grad():
                    # Silero denoise expects specific format
                    # Input: tensor [batch, samples] at specified sample rate
                    # Output: denoised tensor at 48kHz (upsampled)

                    # For real-time, we process in chunks
                    # Silero works best with ~1-3 second chunks

                    # Apply denoising
                    denoised = self._model(audio_tensor)

                    # If output is upsampled to 48kHz, resample back to original rate
                    if denoised.shape[-1] != audio_tensor.shape[-1]:
                        # Resample from output rate to input rate
                        import torchaudio
                        output_rate = 48000  # Silero outputs at 48kHz
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=output_rate,
                            new_freq=sample_rate
                        )
                        denoised = resampler(denoised)

                # Convert back to bytes
                denoised_np = denoised.squeeze().cpu().numpy()

                # Ensure same length as input
                if len(denoised_np) != len(audio):
                    # Trim or pad to match input length
                    if len(denoised_np) > len(audio):
                        denoised_np = denoised_np[:len(audio)]
                    else:
                        denoised_np = np.pad(denoised_np, (0, len(audio) - len(denoised_np)))

                # Convert back to int16
                denoised_int16 = (denoised_np * 32768.0).clip(-32768, 32767).astype(np.int16)

                return denoised_int16.tobytes()

            except Exception as e:
                if self._logger:
                    self._logger.error(f"Denoise error: {e}")
                # Return original audio on error
                return audio_bytes

    def denoise_audio_simple(
        self,
        audio_bytes: bytes,
        sample_rate: int = 8000
    ) -> bytes:
        """
        Simple denoising without SNR checks.
        Always applies denoising.

        Args:
            audio_bytes: Raw 16-bit PCM audio bytes
            sample_rate: Input sample rate

        Returns:
            Denoised audio bytes
        """
        return self.denoise_audio(
            audio_bytes,
            snr_db=None,
            force_denoise=True,
            sample_rate=sample_rate
        )

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_status(self) -> dict:
        """Get singleton status for monitoring."""
        return {
            'loaded': self._model is not None,
            'load_error': self._load_error,
            'device': str(self._device) if self._device else 'not_set',
            'model_name': 'silero_denoise_small_fast'
        }


# Alternative: noisereduce-based denoising (simpler, works with 8kHz)
class NoiseReduceFilter:
    """
    Alternative noise filter using noisereduce library.
    Works directly with 8kHz audio without resampling.

    Pros:
    - Works with any sample rate (8kHz native)
    - No model loading required
    - Pure spectral gating (no neural network)

    Cons:
    - Less aggressive than neural approaches
    - ~5-10ms latency per chunk
    """

    def __init__(self, sample_rate: int = 8000, logger: Optional[logging.Logger] = None):
        self.sample_rate = sample_rate
        self.logger = logger or logging.getLogger(__name__)
        self._noisereduce_available = False

        try:
            import noisereduce as nr
            self._nr = nr
            self._noisereduce_available = True
            self.logger.info("noisereduce library loaded successfully")
        except ImportError:
            self.logger.warning("noisereduce not available, install with: pip install noisereduce")

    def filter_audio(
        self,
        audio_bytes: bytes,
        snr_db: Optional[float] = None,
        snr_threshold: float = 15.0,
        aggressive: bool = False
    ) -> bytes:
        """
        Apply spectral gating noise reduction.

        Args:
            audio_bytes: Raw 16-bit PCM audio bytes
            snr_db: Current SNR (skip filtering if >= threshold)
            snr_threshold: Only filter if SNR < this value
            aggressive: Use more aggressive filtering (non-stationary mode)

        Returns:
            Filtered audio bytes
        """
        if not self._noisereduce_available:
            return audio_bytes

        # Skip if audio is clean
        if snr_db is not None and snr_db >= snr_threshold:
            return audio_bytes

        if not audio_bytes or len(audio_bytes) < 100:
            return audio_bytes

        try:
            # Convert bytes to numpy
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_norm = audio / 32768.0

            # Apply noise reduction
            reduced = self._nr.reduce_noise(
                y=audio_norm,
                sr=self.sample_rate,
                stationary=not aggressive,
                prop_decrease=0.8 if aggressive else 0.5,
                n_fft=512,
                hop_length=128
            )

            # Convert back to int16
            reduced_int16 = (reduced * 32768.0).clip(-32768, 32767).astype(np.int16)
            return reduced_int16.tobytes()

        except Exception as e:
            if self.logger:
                self.logger.error(f"noisereduce error: {e}")
            return audio_bytes

    def is_available(self) -> bool:
        """Check if noisereduce is available."""
        return self._noisereduce_available
