#!/usr/bin/env python3
"""
Chunked Audio Playback System for FreeSWITCH Bot (In-Memory)

Implements on-demand chunked audio playback for fine-grained pause/resume control.
Audio is chunked in memory at playback time - NO files are created or modified.

Key Features:
- On-demand chunking in RAM (no disk writes)
- Pause at any chunk boundary (every 500ms by default)
- Resume from exact paused position
- Uses FreeSWITCH file_string for in-memory playback
- Thread-safe design for concurrent barge-in detection
- Minimal memory footprint (only current chunk in memory)

Architecture:
- ChunkedPlaybackController: Manages playback with pause/resume support
- Audio is read once, split into chunks in memory
- Each chunk played via FreeSWITCH playback command using temp file or file_string
"""

import os
import wave
import struct
import subprocess
import threading
import time
import tempfile
import logging
from enum import Enum
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass

# Import ESL for FreeSWITCH interaction
try:
    import ESL
except ImportError:
    ESL = None
    print("WARNING: ESL module not available - playback will fail")


class ChunkedPlaybackResult(Enum):
    """Result of a chunked playback operation"""
    COMPLETED = "completed"          # All chunks played to completion
    PAUSED = "paused"               # Playback paused at chunk boundary
    INTERRUPTED = "interrupted"      # Barge-in detected during chunk
    FAILED = "failed"               # Error during playback
    CHANNEL_GONE = "channel_gone"   # Channel hung up
    INTENT_DETECTED = "intent_detected"  # Non-neutral intent detected during pause


@dataclass
class AudioChunk:
    """In-memory audio chunk"""
    index: int
    samples: List[int]  # 16-bit PCM samples
    start_sample: int   # Position in original file
    end_sample: int
    duration_ms: float


class ChunkedPlaybackController:
    """
    Manages chunk-based audio playback with pause/resume support.

    Audio is chunked ON-DEMAND in memory - no files are modified.
    Each chunk is played via a temporary file that is immediately deleted.

    Usage:
        controller = ChunkedPlaybackController(conn, uuid, ct_handler, logger)
        result = controller.play_audio("/path/to/audio.wav")

        if result == ChunkedPlaybackResult.PAUSED:
            # Later: controller.resume()
    """

    # Configuration
    DEFAULT_CHUNK_DURATION_MS = 500  # 500ms chunks = pause granularity
    SAMPLE_RATE = 8000               # FreeSWITCH default (8kHz)

    def __init__(
        self,
        conn,
        uuid: str,
        continuous_transcription,
        logger: logging.Logger,
        chunk_duration_ms: float = DEFAULT_CHUNK_DURATION_MS,
        on_chunk_complete: Optional[Callable[[int, int], None]] = None,
        auto_resume_on_silence: bool = False,
        silence_threshold: float = 5.0,
        pause_silence_threshold: float = 0.7,
        on_pause_intent_check: Optional[Callable[[str], Tuple[Optional[str], Optional[str]]]] = None,
        on_interrupt_resume: Optional[Callable[[], bool]] = None,
        interrupt_hard_timeout: float = 5.0
    ):
        """
        Initialize the playback controller.

        Args:
            conn: ESL connection object
            uuid: FreeSWITCH channel UUID
            continuous_transcription: ContinuousTranscriptionHandler instance
            logger: Logger instance
            chunk_duration_ms: Duration of each chunk in milliseconds
            on_chunk_complete: Optional callback(chunk_index, total_chunks) after each chunk
            auto_resume_on_silence: If True, automatically resume after user stops speaking
            silence_threshold: Seconds of silence before auto-resuming (default 5.0)
            pause_silence_threshold: Seconds of silence to trigger intent check during pause (default 0.7)
            on_pause_intent_check: Optional callback(transcription_text) -> (intent, transcription) for pause intent detection
            on_interrupt_resume: Optional callback() -> bool called when resuming after neutral intent.
                                 Returns True to allow resume, False to skip interrupt detection for rest of call.
            interrupt_hard_timeout: Max seconds to wait during interrupt pause before resuming (default 5.0).
                                    Resumes playback even if caller is still speaking.
        """
        self.conn = conn
        self.uuid = uuid
        self.ct = continuous_transcription
        self.logger = logger
        self.chunk_duration_ms = chunk_duration_ms
        self.on_chunk_complete = on_chunk_complete

        # Auto-resume configuration
        self.auto_resume_on_silence = auto_resume_on_silence
        self.silence_threshold = silence_threshold

        # Pause intent detection configuration
        self.pause_silence_threshold = pause_silence_threshold
        self.on_pause_intent_check = on_pause_intent_check

        # Interrupt resume callback - called when resuming after neutral intent
        # Returns True to allow resume, False to disable future interrupts
        self.on_interrupt_resume = on_interrupt_resume

        # Hard timeout for interrupt pause - resumes playback even if caller still speaking
        self.interrupt_hard_timeout = interrupt_hard_timeout

        # Calculate samples per chunk
        self.samples_per_chunk = int(self.SAMPLE_RATE * chunk_duration_ms / 1000)

        # Playback state
        self._state_lock = threading.RLock()
        self._is_playing = False
        self._pause_requested = threading.Event()

        # Current audio state (in-memory)
        self._current_file: Optional[str] = None
        self._chunks: List[AudioChunk] = []
        self._current_chunk_index = 0
        self._paused_at_chunk = 0
        self._sample_rate = self.SAMPLE_RATE

        # Speech-triggered pause/resume state
        self._waiting_for_speech_end = False
        self._speech_paused = threading.Event()  # Set when paused due to speech

        # Detected intent during pause (for handler to retrieve)
        self._detected_intent: Optional[str] = None
        self._detected_transcription: Optional[str] = None

        # Barge-in handling (legacy - kept for compatibility)
        self._barge_in_detected = threading.Event()
        self._callbacks_set = False

        # Track if interrupts were permanently disabled (retries exhausted)
        # This persists across multiple play_audio() calls on same controller
        self._interrupts_permanently_disabled = False

        # Temp file tracking for cleanup
        self._temp_files: List[str] = []
        self._temp_lock = threading.Lock()

    def _setup_speech_callbacks(self):
        """Set up speech detection callbacks for auto pause/resume"""
        # Skip if interrupts were permanently disabled (retries exhausted)
        if self._interrupts_permanently_disabled:
            self.logger.debug("[CHUNKED] Interrupts permanently disabled - skipping callback setup")
            return
        if self.ct and not self._callbacks_set:
            try:
                self.ct.set_speech_callbacks(
                    on_start=self._on_speech_start,
                    on_end=self._on_speech_end,
                    silence_threshold=self.silence_threshold
                )
                self._callbacks_set = True
                self.logger.info(f"[CHUNKED] Speech callbacks configured (auto_resume={self.auto_resume_on_silence}, silence={self.silence_threshold}s)")
            except AttributeError as e:
                self.logger.warning(f"[CHUNKED] ContinuousTranscription doesn't support speech callbacks: {e}")

    def _cleanup_speech_callbacks(self):
        """Clear speech detection callbacks"""
        if self._callbacks_set and self.ct:
            try:
                self.ct.clear_speech_callbacks()
                self.logger.debug("[CHUNKED] Speech callbacks cleared")
            except Exception as e:
                self.logger.warning(f"[CHUNKED] Failed to clear speech callbacks: {e}")
            self._callbacks_set = False

    def _on_speech_start(self):
        """
        Called when user starts speaking during playback.
        Triggers pause at next chunk boundary.
        """
        with self._state_lock:
            if not self._is_playing:
                self.logger.debug("[CHUNKED] Speech detected but not playing - ignoring")
                return
            if self._pause_requested.is_set():
                self.logger.debug("[CHUNKED] Speech detected but already pausing - ignoring")
                return

        self.logger.info("[CHUNKED] Speech detected - pausing playback at next chunk boundary")
        self._waiting_for_speech_end = True
        self._speech_paused.set()
        self._pause_requested.set()

    def _on_speech_end(self):
        """
        Called when user stops speaking (after silence threshold).
        Clears the waiting flag to unblock the playback loop.
        """
        if not self._waiting_for_speech_end:
            self.logger.debug("[CHUNKED] Speech end but not waiting - ignoring")
            return

        self.logger.info(f"[CHUNKED] Silence threshold reached ({self.silence_threshold}s) - signaling resume")

        # Clear the waiting flag - this unblocks the while loop in _play_chunks()
        # which is waiting for silence before resuming playback
        self._waiting_for_speech_end = False
        # Note: _speech_paused and _pause_requested are cleared in _play_chunks() after unblock

    def _auto_resume(self):
        """Resume playback after speech ends (runs in separate thread)"""
        try:
            # Small delay for smoother transition
            time.sleep(0.1)

            # Check if channel is still active before resuming
            if not self._is_channel_active():
                self.logger.warning("[CHUNKED] Channel gone - cannot auto-resume")
                return

            # Resume playback
            self.resume()

        except Exception as e:
            self.logger.error(f"[CHUNKED] Auto-resume error: {e}", exc_info=True)

    def _is_channel_active(self) -> bool:
        """Check if the FreeSWITCH channel is still active"""
        try:
            result = self.conn.api("uuid_exists", self.uuid)
            if result:
                return result.getBody().strip().lower() == "true"
        except Exception as e:
            self.logger.error(f"[CHUNKED] Error checking channel status: {e}")
        return False

    def _load_and_chunk_audio(self, audio_path: str) -> bool:
        """
        Load audio file (any format) and split into chunks in memory.

        Uses ffmpeg to decode any audio format (MP3, WAV, OGG, etc.) to raw PCM.

        Args:
            audio_path: Path to the audio file (any format supported by ffmpeg)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use ffmpeg to decode any audio format to raw 16-bit PCM
            # Output: 8kHz mono (SIP standard) for consistent chunk timing
            result = subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-f', 's16le',           # Raw 16-bit signed little-endian PCM
                '-acodec', 'pcm_s16le',
                '-ar', '8000',           # 8kHz sample rate (SIP standard)
                '-ac', '1',              # Mono
                '-v', 'error',           # Suppress info output
                '-'                      # Output to stdout
            ], capture_output=True, timeout=30)

            if result.returncode != 0:
                self.logger.error(f"[CHUNKED] ffmpeg conversion failed: {result.stderr.decode()}")
                return False

            audio_data = result.stdout
            sample_rate = 8000  # We requested 8kHz from ffmpeg
            self._sample_rate = sample_rate

            # Convert raw bytes to 16-bit samples
            n_samples = len(audio_data) // 2
            if n_samples == 0:
                self.logger.error(f"[CHUNKED] No audio data from ffmpeg for {audio_path}")
                return False

            samples = list(struct.unpack(f'<{n_samples}h', audio_data))

            # Calculate chunk size based on sample rate
            samples_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)

            # Split into chunks
            self._chunks = []
            total_samples = len(samples)
            chunk_index = 0
            start_sample = 0

            while start_sample < total_samples:
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                chunk_samples = samples[start_sample:end_sample]

                duration_ms = (len(chunk_samples) / sample_rate) * 1000

                self._chunks.append(AudioChunk(
                    index=chunk_index,
                    samples=chunk_samples,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    duration_ms=duration_ms
                ))

                start_sample = end_sample
                chunk_index += 1

            self._current_file = audio_path

            self.logger.debug(
                f"[CHUNKED] Loaded {audio_path}: {len(self._chunks)} chunks "
                f"({self.chunk_duration_ms}ms each, {total_samples / sample_rate:.2f}s total)"
            )

            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"[CHUNKED] ffmpeg timeout loading {audio_path}")
            return False
        except Exception as e:
            self.logger.error(f"[CHUNKED] Failed to load audio {audio_path}: {e}", exc_info=True)
            return False

    def _write_temp_wav(self, samples: List[int]) -> Optional[str]:
        """
        Write samples to a temporary WAV file for playback.

        Args:
            samples: 16-bit PCM samples

        Returns:
            Path to temp file, or None on failure
        """
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='chunk_')
            os.close(fd)  # Close the file descriptor, we'll use wave module

            with wave.open(temp_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._sample_rate)
                wav.writeframes(struct.pack(f'<{len(samples)}h', *samples))

            # Make readable by FreeSWITCH (runs as freeswitch user, not root)
            os.chmod(temp_path, 0o644)

            # Track for cleanup
            with self._temp_lock:
                self._temp_files.append(temp_path)

            return temp_path

        except Exception as e:
            self.logger.error(f"[CHUNKED] Failed to write temp WAV: {e}")
            return None

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up a temporary file"""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            with self._temp_lock:
                if temp_path in self._temp_files:
                    self._temp_files.remove(temp_path)
        except Exception as e:
            self.logger.warning(f"[CHUNKED] Failed to cleanup temp file: {e}")

    def _cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        with self._temp_lock:
            files_to_clean = list(self._temp_files)
            self._temp_files.clear()

        for temp_path in files_to_clean:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

    def play_audio(
        self,
        audio_path: str,
        start_chunk: int = 0,
        auto_cleanup: bool = True
    ) -> ChunkedPlaybackResult:
        """
        Play audio file using chunked playback.

        Args:
            audio_path: Full path to audio file
            start_chunk: Starting chunk index (for resume)
            auto_cleanup: Clean up callback after playback

        Returns:
            ChunkedPlaybackResult indicating how playback ended
        """
        # Load and chunk audio if needed (new file or different file)
        if self._current_file != audio_path or not self._chunks:
            if not self._load_and_chunk_audio(audio_path):
                return ChunkedPlaybackResult.FAILED

        if not self._chunks:
            self.logger.error(f"[CHUNKED] No chunks available for: {audio_path}")
            return ChunkedPlaybackResult.FAILED

        # Validate start chunk
        if start_chunk >= len(self._chunks):
            self.logger.warning(f"[CHUNKED] Invalid start chunk {start_chunk}, starting from 0")
            start_chunk = 0

        # Initialize state
        with self._state_lock:
            self._current_chunk_index = start_chunk
            self._paused_at_chunk = start_chunk
            self._is_playing = True
            self._pause_requested.clear()
            self._barge_in_detected.clear()
            self._waiting_for_speech_end = False
            self._speech_paused.clear()

        # Set up speech detection callbacks for auto pause/resume
        self._setup_speech_callbacks()

        try:
            result = self._play_chunks(start_chunk)
            return result

        except Exception as e:
            self.logger.error(f"[CHUNKED] Playback error: {e}", exc_info=True)
            return ChunkedPlaybackResult.FAILED

        finally:
            with self._state_lock:
                self._is_playing = False

            if auto_cleanup:
                self._cleanup_speech_callbacks()
                self._cleanup_all_temp_files()

    def _play_chunks(self, start_chunk: int) -> ChunkedPlaybackResult:
        """
        Play chunks sequentially with pause/resume support.

        Python has control between each chunk, allowing for
        granular pause/resume at chunk boundaries.
        """
        total_chunks = len(self._chunks)

        self.logger.info(
            f"[CHUNKED] Starting playback: {os.path.basename(self._current_file)} "
            f"(chunk {start_chunk + 1}/{total_chunks})"
        )

        i = start_chunk
        while i < total_chunks:
            chunk = self._chunks[i]

            with self._state_lock:
                self._current_chunk_index = i

            # Check for pause request BEFORE playing chunk
            if self._pause_requested.is_set():
                with self._state_lock:
                    self._paused_at_chunk = i

                if self._speech_paused.is_set():
                    # Paused due to user speech - detect 0.7s silence then check intent
                    self.logger.info(f"[CHUNKED] Paused at chunk {i + 1}/{total_chunks} - detecting silence for intent check")

                    # Record when pause started (to get transcriptions since this time)
                    pause_start_time = time.time()

                    # Reset detected intent from previous pause
                    self._detected_intent = None
                    self._detected_transcription = None

                    # CRITICAL: We're paused because speech was detected, so set this to True!
                    # If user already stopped speaking, silence_start will be set immediately
                    speech_detected = True
                    silence_start = None

                    # Use 300ms FS sleep intervals on main thread
                    # ESL is NOT thread-safe (background threads cause segfaults)
                    # Longer intervals = fewer commands = less timing corruption
                    while True:
                        try:
                            self.conn.execute("sleep", "300")
                        except Exception as e:
                            self.logger.warning(f"[CHUNKED] Keepalive failed: {e}")
                            time.sleep(0.3)

                        # Check hard timeout - resume even if caller still speaking
                        pause_elapsed = time.time() - pause_start_time
                        if pause_elapsed >= self.interrupt_hard_timeout:
                            self.logger.warning(
                                f"[CHUNKED] Interrupt hard timeout ({self.interrupt_hard_timeout}s) reached - resuming playback")
                            break

                        # Check channel still active
                        if not self._is_channel_active():
                            self.logger.warning("[CHUNKED] Channel gone during pause")
                            return ChunkedPlaybackResult.CHANNEL_GONE

                        # Check speech state from continuous transcription handler
                        if self.ct:
                            try:
                                with self.ct.speech_lock:
                                    is_speaking = self.ct.is_speaking
                            except Exception:
                                is_speaking = False

                            if is_speaking:
                                # User is still speaking - reset silence tracking
                                silence_start = None
                            else:
                                # Silence - start or continue tracking
                                if silence_start is None:
                                    silence_start = time.time()
                                    self.logger.debug(f"[CHUNKED] Speech ended, silence started")

                                # Check if 0.7s silence reached
                                silence_duration = time.time() - silence_start
                                if silence_duration >= self.pause_silence_threshold:
                                    self.logger.info(f"[CHUNKED] {self.pause_silence_threshold}s silence detected - checking intent")
                                    break

                    # Grace period: let RTP timer settle after sleep commands
                    time.sleep(0.3)
                    self.logger.debug("[CHUNKED] Pause loop exited with grace period")

                    # Get EXISTING transcriptions from CT (already transcribed during playback!)
                    # No need to re-transcribe - use what CT already captured
                    if self.on_pause_intent_check and self.ct:
                        # Get transcriptions from last 5 seconds (covers the triggering phrase)
                        try:
                            recent_transcriptions = self.ct.get_transcriptions_since(
                                pause_start_time - 5.0,  # Include transcriptions from before pause
                                min_confidence=0.3
                            )
                        except Exception as e:
                            self.logger.warning(f"[CHUNKED] Error getting transcriptions: {e}")
                            recent_transcriptions = []

                        if recent_transcriptions:
                            # Use only the LAST transcription - it's the most complete
                            # (CT does incremental transcription, each one fuller than previous)
                            _, combined_text, _ = recent_transcriptions[-1]
                            self.logger.info(f"[CHUNKED] Using latest transcription: '{combined_text}'")

                            try:
                                intent, transcription = self.on_pause_intent_check(combined_text)

                                if intent and intent not in ("neutral", "unhandled_question"):
                                    # Non-neutral intent - return to handler for routing
                                    self.logger.info(f"[CHUNKED] Intent detected: {intent} - stopping playback")
                                    self._detected_intent = intent
                                    self._detected_transcription = transcription or combined_text
                                    return ChunkedPlaybackResult.INTENT_DETECTED
                                else:
                                    # Neutral or no intent - resume playback
                                    self.logger.info(f"[CHUNKED] Neutral/no intent - resuming from chunk {i + 1}")
                                    # Notify handler that an interrupt retry was used
                                    if self.on_interrupt_resume:
                                        try:
                                            allow_future = self.on_interrupt_resume()
                                            if not allow_future:
                                                # Retries exhausted - disable speech callbacks permanently
                                                self.logger.warning("[CHUNKED] Interrupt retries exhausted - disabling future interrupts")
                                                self._interrupts_permanently_disabled = True
                                                self._cleanup_speech_callbacks()
                                        except Exception as e:
                                            self.logger.error(f"[CHUNKED] on_interrupt_resume callback failed: {e}")
                            except Exception as e:
                                self.logger.error(f"[CHUNKED] Intent check callback failed: {e}")
                                # On error, default to resume
                                self.logger.info(f"[CHUNKED] Defaulting to resume after error")
                        else:
                            self.logger.info(f"[CHUNKED] No transcription found - resuming from chunk {i + 1}")

                    # Clear CT buffer after processing to avoid duplicate transcription
                    if self.ct:
                        try:
                            self.ct.clear_buffer()
                        except Exception as e:
                            self.logger.warning(f"[CHUNKED] Error clearing CT buffer: {e}")

                    # Clear pause flags and continue
                    self._pause_requested.clear()
                    self._speech_paused.clear()
                    self._waiting_for_speech_end = False
                    continue  # Re-process this chunk (don't increment i)

                elif self._barge_in_detected.is_set():
                    self.logger.info(f"[CHUNKED] Interrupted at chunk {i + 1}/{total_chunks} (barge-in)")
                    return ChunkedPlaybackResult.INTERRUPTED
                else:
                    self.logger.info(f"[CHUNKED] Paused at chunk {i + 1}/{total_chunks}")
                    return ChunkedPlaybackResult.PAUSED

            # Check channel is still active
            if not self._is_channel_active():
                self.logger.warning("[CHUNKED] Channel gone - stopping playback")
                return ChunkedPlaybackResult.CHANNEL_GONE

            # Write chunk to temp file
            temp_path = self._write_temp_wav(chunk.samples)
            if temp_path is None:
                self.logger.error(f"[CHUNKED] Failed to create temp file for chunk {i}")
                return ChunkedPlaybackResult.FAILED

            try:
                # Play this chunk (BLOCKING)
                self.logger.debug(f"[CHUNKED] Playing chunk {i + 1}/{total_chunks} ({chunk.duration_ms:.0f}ms)")
                self.conn.execute("playback", temp_path)

            finally:
                # Clean up temp file immediately after playback
                self._cleanup_temp_file(temp_path)

            # Optional callback after chunk completes
            if self.on_chunk_complete:
                try:
                    self.on_chunk_complete(i, total_chunks)
                except Exception as e:
                    self.logger.warning(f"[CHUNKED] Chunk callback error: {e}")

            # Brief yield to allow pause requests to be processed
            time.sleep(0.001)

            # Move to next chunk
            i += 1

        self.logger.info(f"[CHUNKED] Playback completed: {os.path.basename(self._current_file)}")
        return ChunkedPlaybackResult.COMPLETED

    def pause(self):
        """
        Request pause at next chunk boundary.

        Non-blocking - playback will pause after the current chunk
        finishes (max latency = chunk_duration_ms).
        """
        self.logger.info("[CHUNKED] Pause requested")
        self._pause_requested.set()

    def resume(self) -> ChunkedPlaybackResult:
        """
        Resume playback from the paused chunk.

        Returns:
            ChunkedPlaybackResult from resumed playback
        """
        with self._state_lock:
            if self._current_file is None or not self._chunks:
                self.logger.warning("[CHUNKED] Cannot resume - no current file")
                return ChunkedPlaybackResult.FAILED

            start_chunk = self._paused_at_chunk

            # Clear pause state
            self._pause_requested.clear()
            self._barge_in_detected.clear()
            self._speech_paused.clear()
            self._waiting_for_speech_end = False

        self.logger.info(f"[CHUNKED] Resuming from chunk {start_chunk + 1}/{len(self._chunks)}")
        return self.play_audio(self._current_file, start_chunk=start_chunk)

    def reset(self):
        """Reset controller state for new playback"""
        with self._state_lock:
            self._current_file = None
            self._chunks = []
            self._current_chunk_index = 0
            self._paused_at_chunk = 0
            self._pause_requested.clear()
            self._barge_in_detected.clear()
            self._waiting_for_speech_end = False
            self._speech_paused.clear()

        self._cleanup_speech_callbacks()
        self._cleanup_all_temp_files()

    @property
    def is_playing(self) -> bool:
        """Check if currently playing"""
        with self._state_lock:
            return self._is_playing

    @property
    def is_paused(self) -> bool:
        """Check if playback is paused"""
        with self._state_lock:
            return self._pause_requested.is_set() and not self._is_playing

    @property
    def current_position_ms(self) -> float:
        """Get approximate current position in milliseconds"""
        with self._state_lock:
            if not self._chunks:
                return 0.0

            if self._current_chunk_index >= len(self._chunks):
                # Return total duration
                return sum(c.duration_ms for c in self._chunks)

            # Sum duration of completed chunks
            return sum(c.duration_ms for c in self._chunks[:self._current_chunk_index])

    @property
    def total_duration_ms(self) -> float:
        """Get total audio duration in milliseconds"""
        with self._state_lock:
            return sum(c.duration_ms for c in self._chunks)

    @property
    def progress_percent(self) -> float:
        """Get playback progress as percentage (0-100)"""
        with self._state_lock:
            if not self._chunks:
                return 0.0

            total = len(self._chunks)
            return (self._current_chunk_index / total) * 100

    @property
    def paused_chunk_index(self) -> int:
        """Get the chunk index where playback was paused"""
        with self._state_lock:
            return self._paused_at_chunk

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks"""
        with self._state_lock:
            return len(self._chunks)

    @property
    def remaining_chunks(self) -> int:
        """Get number of chunks remaining"""
        with self._state_lock:
            return len(self._chunks) - self._current_chunk_index

    @property
    def detected_intent(self) -> Optional[str]:
        """Get the intent detected during pause (if any)"""
        return self._detected_intent

    @property
    def detected_transcription(self) -> Optional[str]:
        """Get the transcription from pause intent detection (if any)"""
        return self._detected_transcription
