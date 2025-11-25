#!/usr/bin/env python3
"""
Continuous Transcription Handler for FreeSWITCH Bot
- Handles real-time transcription during bot playback
- Detects DNC/NI keywords using intent detector
- Manages audio buffering and transcription queue
- Thread-safe design for concurrent audio processing
"""

import threading
import time
import os
import logging
import tempfile
import numpy as np
from typing import Optional, List, Tuple, Callable
from collections import deque

# Import Silero VAD singleton for speech detection
from src.silero_vad_singleton import SileroVADSingleton


class ContinuousTranscriptionHandler:
    """
    Handles continuous transcription of caller audio during bot playback.
    Detects DNC/NI intents in real-time without interrupting playback.
    """

    def __init__(self, parakeet_model, intent_detector, logger, rnnt_confidence_threshold=0.3, energy_threshold=0.045, immediate_hangup_callback=None):
        """
        Initialize continuous transcription handler

        Args:
            parakeet_model: Parakeet RNNT model for transcription
            intent_detector: IntentDetector for keyword matching
            logger: Logger instance
            rnnt_confidence_threshold: Minimum confidence for transcriptions
            energy_threshold: Silero VAD threshold from database (e_campaign.energy_threshold)
            immediate_hangup_callback: Optional callback function for immediate hangup on DNC/NI/HP detection
                                      Signature: callback(disposition: str, intent_type: str)
        """
        self.parakeet_model = parakeet_model
        self.intent_detector = intent_detector
        self.logger = logger
        self.rnnt_confidence_threshold = rnnt_confidence_threshold
        self.energy_threshold = energy_threshold
        self.immediate_hangup_callback = immediate_hangup_callback

        # Get Silero VAD singleton reference (preloaded by bot_server at startup)
        # Store threshold for use in inference calls
        self.silero_vad_singleton = SileroVADSingleton()
        self.logger.info(f"Silero VAD singleton ready for continuous transcription (threshold={energy_threshold})")

        # Speech-silence pattern detection (from old continuous_listener.py)
        self.min_speech_duration = 0.08  # 80ms minimum speech
        self.silence_threshold = 1.0  # 1 second of silence to trigger transcription
        self.min_audio_for_processing = 0.15  # 150ms minimum audio to process
        self.speech_pad_ms = 30  # Padding for speech segments

        # Speech detection state
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.speech_lock = threading.Lock()

        # Audio buffer management (rolling buffer for speech segments)
        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.buffer_max_duration = 120.0  # Keep up to 120 seconds for context (matches pjsua2)
        self.buffer_min_duration = 2.0  # Minimum 2 seconds before transcribing

        # Transcription storage
        self.transcriptions = []  # List of (timestamp, text, confidence) tuples
        self.transcription_lock = threading.Lock()
        self.max_transcription_age = 120.0  # Keep last 120 seconds

        # Playback tracking
        self.current_playback_start = None
        self.current_playback_end = None
        self.playback_periods = []  # Track (start_time, end_time) tuples for race condition fix
        self.playback_lock = threading.Lock()

        # Detection flags
        self.dnc_detected = False
        self.ni_detected = False
        self.hp_detected = False  # Hold/Press honeypot detection
        self.audio_activity_detected = False  # For DAIR vs DAIR 2 disposition
        self.detection_lock = threading.Lock()

        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'transcriptions_attempted': 0,
            'transcriptions_successful': 0,
            'dnc_detections': 0,
            'ni_detections': 0,
            'hp_detections': 0,  # Honeypot detections
            'vad_speech_segments': 0,
            'vad_silence_segments': 0
        }

        self.logger.info("ContinuousTranscriptionHandler initialized with Silero VAD")

    def mark_playback_start(self):
        """Mark that bot audio playback has started"""
        with self.playback_lock:
            self.current_playback_start = time.time()
            self.current_playback_end = None

    def mark_playback_end(self):
        """Mark that bot audio playback has ended"""
        with self.playback_lock:
            if self.current_playback_start:
                self.current_playback_end = time.time()

                # Store playback period for race condition handling
                self.playback_periods.append((self.current_playback_start, self.current_playback_end))

                # Trim old playback periods (keep last 10 seconds)
                cutoff_time = time.time() - 10.0
                self.playback_periods = [(s, e) for s, e in self.playback_periods if e > cutoff_time]

                self.current_playback_start = None

    def is_during_playback(self) -> bool:
        """
        Check if bot is currently playing audio or if we're in the window
        after playback (to handle delayed audio chunks from file I/O)
        """
        with self.playback_lock:
            # Quick check: Currently playing
            if self.current_playback_start is not None:
                return True

            # Check recent playback periods (handles race condition with file I/O)
            if self.playback_periods:
                current_time = time.time()
                for start, end in self.playback_periods[-3:]:  # Check last 3 playbacks
                    # Allow 2-second window after playback for delayed chunks
                    if start <= current_time <= (end + 2.0):
                        return True

            return False

    def add_audio_chunk(self, audio_bytes: bytes):
        """
        Add audio chunk to buffer and update speech detection state

        Args:
            audio_bytes: Raw audio bytes (8kHz, 16-bit mono PCM)
        """
        with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            self.stats['chunks_processed'] += 1

            # Trim buffer if too large (keep last N seconds)
            sample_rate = 8000
            bytes_per_second = sample_rate * 2  # 16-bit = 2 bytes per sample
            max_bytes = int(self.buffer_max_duration * bytes_per_second)

            if len(self.audio_buffer) > max_bytes:
                # Keep only the most recent audio
                self.audio_buffer = bytearray(self.audio_buffer[-max_bytes:])

        # Mark that audio activity occurred (for DAIR 2 detection)
        # Check for actual audio energy, not just non-empty bytes
        if len(audio_bytes) > 0 and not self.audio_activity_detected:
            import numpy as np
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(float)**2))

            # Threshold: 50 detects any real audio activity (noise, breathing, static, speech)
            # Filters out complete silence or near-zero samples
            if rms > 50:
                self.logger.info(f"Audio activity detected (RMS={rms:.1f}) - will use DAIR 2 disposition if max silences reached")
                self.audio_activity_detected = True

        # Update speech detection state using Silero VAD
        self._update_speech_state(audio_bytes)

    def should_transcribe(self) -> bool:
        """
        Check if buffer has enough audio to transcribe

        Returns:
            True if ready to transcribe
        """
        with self.buffer_lock:
            sample_rate = 8000
            bytes_per_second = sample_rate * 2
            min_bytes = int(self.buffer_min_duration * bytes_per_second)
            return len(self.audio_buffer) >= min_bytes

    def get_audio_for_transcription(self) -> Optional[bytes]:
        """
        Get audio from buffer for transcription (last 2-3 seconds)

        Returns:
            Audio bytes or None if insufficient data
        """
        with self.buffer_lock:
            if not self.should_transcribe():
                return None

            # Return copy of buffer (last 2-3 seconds)
            return bytes(self.audio_buffer)

    def transcribe_and_check_intents(self):
        """
        Transcribe buffered audio and check for DNC/NI intents
        Uses DUAL-MODE processing (matching old continuous_listener.py):
        - MODE 1 (DURING playback): Immediate transcription when speech detected
        - MODE 2 (AFTER playback): Wait for speech-silence pattern

        This is called from the unified detection loop for each audio chunk (every 50ms)

        NOTE: The handler already checks playback state before calling this method,
        so we don't need to check is_during_playback() here. This fixes the state
        synchronization bug where the handler's playback state and transcription
        handler's internal state could be out of sync.
        """
        # REMOVED: is_during_playback() check - handler already verified playback state
        # This fixes the state synchronization bug between dual tracking systems

        # === MODE 1: IMMEDIATE PROCESSING DURING PLAYBACK ===
        # (Matches old continuous_listener.py lines 339-349)
        # Transcribe immediately when we detect speech, no silence waiting required

        # Check if we have recent speech in buffer
        with self.speech_lock:
            has_recent_speech = self.is_speaking or (
                not self.is_speaking and
                self.silence_start_time and
                (time.time() - self.silence_start_time) < 2.0  # Within 2.0s of speech ending (relaxed from 0.5s)
            )

        if has_recent_speech:
            # Get buffer size
            with self.buffer_lock:
                buffer_size = len(self.audio_buffer)

            # Need at least 0.3 seconds of audio (4800 bytes at 8kHz 16-bit) - reduced from 1.0s (16000 bytes)
            min_buffer_bytes = 4800  # 0.3s at 8kHz 16-bit

            if buffer_size >= min_buffer_bytes:
                # Pull last 3 seconds from buffer for transcription (matches pjsua2)
                sample_rate = 8000
                bytes_per_second = sample_rate * 2
                max_bytes = int(3.0 * bytes_per_second)  # 3 seconds

                with self.buffer_lock:
                    # Get last 3 seconds (or all if less than 3s)
                    if len(self.audio_buffer) > max_bytes:
                        recent_audio = bytes(self.audio_buffer[-max_bytes:])
                    else:
                        recent_audio = bytes(self.audio_buffer)

                if not recent_audio or len(recent_audio) < 1600:  # At least 100ms
                    return

                # Check duration
                audio_duration = len(recent_audio) / bytes_per_second

                if audio_duration < self.min_audio_for_processing:
                    return

                try:
                    self.stats['transcriptions_attempted'] += 1
                    self.logger.info(f"🎙️ [TRANSCRIPTION] Starting immediate transcription ({len(recent_audio)} bytes, {audio_duration:.2f}s)")

                    # Transcribe audio immediately
                    text, confidence = self._transcribe_audio_chunk(recent_audio)

                    if text and len(text.strip()) > 0:
                        self.logger.info(f"✅ [TRANSCRIPTION SUCCESS] '{text}' (confidence: {confidence:.2f})")
                        self.stats['transcriptions_successful'] += 1

                        # Store transcription with timestamp
                        current_time = time.time()
                        with self.transcription_lock:
                            self.transcriptions.append((current_time, text, confidence))

                            # Trim old transcriptions
                            cutoff_time = current_time - self.max_transcription_age
                            self.transcriptions = [
                                (t, txt, conf) for t, txt, conf in self.transcriptions
                                if t > cutoff_time
                            ]

                        # Log transcription
                        self.logger.info(f"🎤 IMMEDIATE (during playback): '{text}' (conf: {confidence:.2f})")

                        # Check for DNC/NI intents using keyword matching
                        self._check_for_intents(text, confidence)

                        # Reset silence timer to prevent duplicate transcription
                        with self.speech_lock:
                            if self.silence_start_time:
                                self.silence_start_time = None

                except Exception as e:
                    self.logger.error(f"Error in immediate transcription: {e}", exc_info=True)

    def _update_speech_state(self, audio_bytes: bytes):
        """
        Update speech detection state using Silero VAD
        Implements speech-silence pattern detection from old continuous_listener.py

        Args:
            audio_bytes: Raw audio bytes (8kHz, 16-bit mono PCM)
        """
        try:
            # Use Silero VAD singleton to detect speech
            is_speech = self.silero_vad_singleton.is_speech(
                audio_bytes,
                threshold=self.energy_threshold,
                sample_rate=8000
            )

            current_time = time.time()

            with self.speech_lock:
                if is_speech:
                    # Speech detected
                    if not self.is_speaking:
                        # Transition from silence to speech
                        self.is_speaking = True
                        self.speech_start_time = current_time
                        self.silence_start_time = None
                        self.stats['vad_speech_segments'] += 1
                        self.logger.debug("🗣️ Speech started")
                else:
                    # Silence detected
                    if self.is_speaking:
                        # Check if we've had minimum speech duration
                        if self.speech_start_time:
                            speech_duration = current_time - self.speech_start_time

                            if speech_duration >= self.min_speech_duration:
                                # Valid speech segment ended, start silence timer
                                self.is_speaking = False
                                self.silence_start_time = current_time
                                self.stats['vad_silence_segments'] += 1
                                self.logger.debug(f"🤐 Speech ended (duration: {speech_duration:.2f}s), silence started")
                            else:
                                # Speech too short, might be noise
                                self.is_speaking = False
                                self.speech_start_time = None

        except Exception as e:
            self.logger.error(f"Error updating speech state: {e}", exc_info=True)

    def _transcribe_audio_chunk(self, audio_bytes: bytes) -> Tuple[Optional[str], float]:
        """
        Transcribe audio chunk using Parakeet

        Args:
            audio_bytes: Raw audio bytes (8kHz, 16-bit mono PCM)

        Returns:
            Tuple of (text, confidence)
        """
        try:
            if not self.parakeet_model or not audio_bytes or len(audio_bytes) < 1600:
                return None, 0.0

            # Import processing libraries
            import torch
            import soundfile as sf
            import scipy.signal as scipy_signal
            import torchaudio

            # Convert to numpy array - use float64 for processing
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)

            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0

            # Apply pre-emphasis filter
            pre_emphasis = 0.97
            audio_array = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1]).copy()

            # Apply bandpass filter (300Hz - 3400Hz for speech)
            nyquist = 4000  # Half of 8kHz sample rate
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = scipy_signal.butter(5, [low, high], btype='band')
            audio_array = scipy_signal.filtfilt(b, a, audio_array).copy()

            # Normalize audio level
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array * (0.9 / max_val)

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).to(torch.float64).unsqueeze(0)

            # Resample from 8kHz to 16kHz
            audio_tensor_32 = audio_tensor.to(torch.float32)
            resampler = torchaudio.transforms.Resample(
                orig_freq=8000,
                new_freq=16000,
                resampling_method='sinc_interp_hann'
            )
            audio_16k = resampler(audio_tensor_32)

            # Save to temporary file
            audio_16k_save = audio_16k.squeeze().numpy().astype(np.float32)

            # Normalize again
            max_val = np.max(np.abs(audio_16k_save))
            if max_val > 0:
                audio_16k_save = audio_16k_save * (0.95 / max_val)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_16k_save, 16000, subtype='PCM_16')
                tmp_path = tmp_file.name

            try:
                # Transcribe with Parakeet
                with torch.no_grad():
                    text, confidence = self.parakeet_model.transcribe_with_confidence(
                        tmp_path,
                        batch_size=1,
                        num_workers=0,
                        verbose=False
                    )

                    # Apply confidence threshold
                    if confidence < self.rnnt_confidence_threshold:
                        self.logger.debug(f"Low confidence ({confidence:.3f}): {text}")
                        return None, confidence

                    # Clean up text
                    if text:
                        text = text.strip()
                        # Remove common ASR artifacts
                        import re
                        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
                        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace

                        # Filter very short transcriptions
                        if len(text.strip()) < 2:
                            return None, confidence

                    return text, confidence

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            self.logger.debug(f"Audio chunk transcription error: {e}")
            return None, 0.0

    def _check_for_intents(self, text: str, confidence: float):
        """
        Check transcribed text for DNC/NI keywords

        Args:
            text: Transcribed text
            confidence: Transcription confidence
        """
        try:
            # Use intent detector for keyword matching
            intent_result = self.intent_detector.detect_intent(text)

            if intent_result:
                intent_type, intent_confidence = intent_result
                self.logger.warning(f"🚫 INTENT DETECTED (continuous): {intent_type} - '{text}' (conf: {intent_confidence:.2f})")

                # Set appropriate flags and determine disposition
                disposition = None
                with self.detection_lock:
                    if intent_type == "do_not_call":
                        self.dnc_detected = True
                        self.stats['dnc_detections'] += 1
                        self.logger.warning(f"🚫 DNC DETECTED during playback!")
                        disposition = "DNC"
                    elif intent_type == "not_interested":
                        self.ni_detected = True
                        self.stats['ni_detections'] += 1
                        self.logger.warning(f"⚠️ NI DETECTED during playback!")
                        disposition = "NI"
                    elif intent_type == "hold_press":
                        self.hp_detected = True
                        self.stats['hp_detections'] += 1
                        self.logger.warning(f"🍯 HP (Hold/Press) DETECTED during playback!")
                        disposition = "HP"
                    elif intent_type == "obscenity":
                        # Treat obscenity as DNC
                        self.dnc_detected = True
                        self.stats['dnc_detections'] += 1
                        self.logger.warning(f"🚫 OBSCENITY DETECTED during playback (treated as DNC)!")
                        disposition = "DNC"

                # Trigger immediate hangup callback if configured
                if disposition and self.immediate_hangup_callback:
                    try:
                        self.immediate_hangup_callback(disposition, intent_type)
                    except Exception as e:
                        self.logger.error(f"Immediate hangup callback failed: {e}", exc_info=True)
                        # Continue - flags are already set, main thread will handle fallback

        except Exception as e:
            self.logger.error(f"Error checking intents: {e}", exc_info=True)

    def get_transcriptions_since(self, start_time: float, min_confidence: float = 0.3) -> List[Tuple[float, str, float]]:
        """
        Get all transcriptions that occurred after start_time

        Args:
            start_time: Unix timestamp to filter from
            min_confidence: Minimum confidence threshold

        Returns:
            List of (timestamp, text, confidence) tuples
        """
        with self.transcription_lock:
            filtered = [
                (timestamp, text, confidence)
                for timestamp, text, confidence in self.transcriptions
                if timestamp >= start_time and confidence >= min_confidence
            ]
            return filtered

    def has_dnc_ni_detection(self) -> Tuple[bool, Optional[str]]:
        """
        Check if DNC, NI, or HP was detected

        Returns:
            Tuple of (detected, intent_type) where intent_type is "DNC", "NI", or "HP"
        """
        with self.detection_lock:
            if self.dnc_detected:
                return True, "DNC"
            elif self.ni_detected:
                return True, "NI"
            elif self.hp_detected:
                return True, "HP"
            return False, None

    def reset_detection_flags(self):
        """Reset DNC/NI/HP detection flags"""
        with self.detection_lock:
            self.dnc_detected = False
            self.ni_detected = False
            self.hp_detected = False

    def get_stats(self) -> dict:
        """Get statistics"""
        return self.stats.copy()

    def has_audio_activity(self) -> bool:
        """Check if any audio was detected during the call (for DAIR vs DAIR 2)"""
        return self.audio_activity_detected

    def clear_buffer(self):
        """Clear audio buffer (call after playback ends to prevent stale audio)"""
        with self.buffer_lock:
            self.audio_buffer.clear()
