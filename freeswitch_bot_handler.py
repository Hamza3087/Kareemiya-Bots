#!/usr/bin/env python3
"""
FreeSWITCH Bot Handler
- Core call handling logic (replaces SuiteCRMBotInstance)
- Plain Python class (no pjsua2 inheritance)
- FreeSWITCH handles SIP/RTP/media
- Python handles business logic only
"""

import os
import time
import tempfile
import logging
import threading
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import ESL
except ImportError:
    print("ERROR: ESL module not installed")
    raise

# Import business logic components (all reused from existing codebase)
from src.call_flow import parse_call_flow_from_string, get_audio_path_for_agent
from src.intent_detector import IntentDetector
from src.suitecrm_integration import SuiteCRMAgentConfig, SuiteCRMLogger
from src.call_outcome_handler import CallOutcomeHandler, CallOutcome
from src.vicidial_integration import ViciDialAPI, VoicebotViciDialIntegration
from src.parakeet_rnnt import ParakeetRNNTModel
from src import config

# Import continuous transcription handler
from src.continuous_transcription_handler import ContinuousTranscriptionHandler

# Import ESL config reader for dynamic configuration
from src.esl_config_reader import ESLConfigReader


# Import Silero VAD singleton for speech detection
from src.silero_vad_singleton import SileroVADSingleton

# Ringing detection components (framework-agnostic)
from src.ringing_detector_core import (
    GoertzelDetector,
    RingCycleTracker,
    DetectionValidator
)
import numpy as np

# Import model singletons (standalone modules)
from src.parakeet_singleton import ParakeetModelSingleton
try:
    from src.qwen_singleton import QwenModelSingleton
except ImportError:
    QwenModelSingleton = None

# Import FreeSWITCH transfer manager
from src.freeswitch_transfer_manager import FreeSWITCHTransferManager

# Import chunked playback controller for pause/resume support
from src.chunked_playback import ChunkedPlaybackController, ChunkedPlaybackResult

# Configuration
SUITECRM_UPLOAD_DIR = "/var/www/recordings"


class ResourceManager:
    """
    Manages cleanup of all call resources
    Pattern from deprecated pjsua2 code (suitecrm_bot_instance.py lines 295-394)
    Ensures all resources are properly freed to prevent leaks
    """
    def __init__(self, logger):
        self.lock = threading.RLock()
        self.resources = {}  # name -> {resource, cleanup, registered}
        self._cleanup_done = False
        self.logger = logger

    def register(self, name, resource, cleanup_func=None):
        """Register resource with cleanup callback"""
        with self.lock:
            if self._cleanup_done:
                return False
            self.resources[name] = {
                'resource': resource,
                'cleanup': cleanup_func,
                'registered': time.time()
            }
            self.logger.debug(f"[RESOURCE] Registered: {name}")
            return True

    def cleanup_all(self):
        """Cleanup all resources in reverse registration order"""
        with self.lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            resources_to_clean = list(self.resources.items())

        # Cleanup in reverse order (last registered, first cleaned)
        for name, entry in reversed(resources_to_clean):
            if entry.get('cleanup'):
                try:
                    entry['cleanup']()
                    self.logger.info(f"[RESOURCE] Cleaned up: {name}")
                except Exception as e:
                    self.logger.error(f"[RESOURCE] Cleanup failed for {name}: {e}")

        with self.lock:
            self.resources.clear()


class CallLogCollector:
    """Thread-safe collector for detailed call logs to be stored in database"""
    def __init__(self, logger, call_id):
        self._lock = threading.RLock()
        self._logs = []
        self._db_metrics = []  # Track database performance metrics
        self._start_time = time.time()
        self._agent_id = getattr(logger, 'name', 'unknown')
        self._call_id = call_id

    def _strip_emojis(self, text: str) -> str:
        """Remove emoji characters that cause MySQL encoding issues"""
        # Remove common emoji patterns and replace with simple text
        emoji_replacements = {
            '🎙️': '[MIC]',
            '🎧': '[HEADPHONE]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🚀': '[ROCKET]',
            '🔌': '[PLUG]',
            '🛑': '[STOP]',
            '💿': '[DISC]',
            '📻': '[RADIO]',
            '🔔': '[BELL]',
            '🚫': '[BLOCKED]',
            '👤': '[PERSON]',
            '🎵': '[MUSIC]',
            '🔍': '[SEARCH]',
            '🌐': '[GLOBE]',
        }

        clean_text = text
        for emoji, replacement in emoji_replacements.items():
            clean_text = clean_text.replace(emoji, replacement)

        # Remove any remaining emoji characters using regex
        import re
        # Unicode ranges for emojis and symbols
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", flags=re.UNICODE)

        return emoji_pattern.sub('', clean_text)

    def add_log(self, level: str, message: str):
        """Add a log entry with timestamp, thread-safe"""
        with self._lock:
            # Strip emojis from the message to prevent database encoding issues
            clean_message = self._strip_emojis(message)
            timestamp = time.time() - self._start_time
            formatted_time = time.strftime('%H:%M:%S', time.localtime(self._start_time + timestamp))
            log_entry = f"[{formatted_time}] [{level}] [{self._agent_id}] [Call-{self._call_id}] {clean_message}"
            self._logs.append(log_entry)

            # Track database operations for metrics
            if any(keyword in clean_message for keyword in ["Query:", "SLOW QUERY:", "Query error", "Query failed"]):
                self.add_db_metric(clean_message)

    def add_db_metric(self, metric_message: str):
        """Add database performance metric with thread safety"""
        with self._lock:
            timestamp = time.time() - self._start_time
            formatted_time = time.strftime('%H:%M:%S', time.localtime(self._start_time + timestamp))
            db_entry = f"[{formatted_time}] {metric_message}"
            self._db_metrics.append(db_entry)

    def get_logs(self) -> str:
        """Get all collected logs as formatted text with database metrics"""
        with self._lock:
            combined = '\n'.join(self._logs)

            # Add database metrics section if we have any
            if self._db_metrics:
                combined += "\n\n=== DATABASE PERFORMANCE METRICS ===\n"
                combined += "\n".join(self._db_metrics)

                # Add summary statistics
                query_count = len([m for m in self._db_metrics if "Query:" in m])
                slow_query_count = len([m for m in self._db_metrics if "SLOW QUERY:" in m])
                error_count = len([m for m in self._db_metrics if "Query error" in m or "Query failed" in m])

                combined += f"\n\n=== DATABASE METRICS SUMMARY ===\n"
                combined += f"Total Queries: {query_count}\n"
                combined += f"Slow Queries (>1s): {slow_query_count}\n"
                combined += f"Query Errors: {error_count}"

            return combined

    def clear(self):
        """Clear all collected logs and database metrics"""
        with self._lock:
            self._logs.clear()
            self._db_metrics.clear()


class FreeSWITCHBotHandler:
    """
    Core call handler for FreeSWITCH ESL
    Replaces the complex pjsua2 SuiteCRMBotInstance with clean ESL-based logic
    """

    def __init__(self, conn: ESL.ESLconnection, agent_config: SuiteCRMAgentConfig, info: ESL.ESLevent,
                 vici_lead_id: Optional[str] = None,
                 vici_list_id: Optional[str] = None,
                 vici_campaign_id: Optional[str] = None):
        """
        Initialize bot handler

        Args:
            conn: ESL connection object
            agent_config: Agent configuration from database
            info: Call information from FreeSWITCH
            vici_lead_id: ViciDial lead ID from SIP headers (optional)
            vici_list_id: ViciDial list ID from SIP headers (optional)
            vici_campaign_id: ViciDial campaign ID from SIP headers (optional)
        """
        self.conn = conn
        self.agent_config = agent_config
        self.info = info

        # ViciDial metadata from SIP headers
        self.vici_lead_id = vici_lead_id
        self.vici_list_id = vici_list_id
        self.vici_campaign_id = vici_campaign_id

        # Extract call information
        self.uuid = info.getHeader("Unique-ID")
        self.phone_number = info.getHeader("Caller-Caller-ID-Number") or "UNKNOWN"
        self.destination = info.getHeader("Caller-Destination-Number")
        self.caller_state = None  # Can be extracted from area code if needed

        # Setup logging
        self.logger = logging.getLogger(f"Call-{self.uuid[:8]}")
        self.logger.setLevel(logging.DEBUG)  # Ensure all log levels are captured

        # Call log collector for database audit trail
        self.call_log_collector = CallLogCollector(self.logger, self.uuid[:8])

        # Resource manager for cleanup tracking (prevents leaks)
        self.resource_manager = ResourceManager(self.logger)

        # Call state
        self.is_active = True
        self.call_start_time = time.time()
        self.current_step = None
        self.call_flow = None
        self.crm_updated = False  # Track if CRM has been updated (for cleanup fallback)

        # Tracking
        self.consecutive_silence_count = 0
        self.clarification_count = 0
        self.conversation_log = []

        # Configuration from agent
        self.max_consecutive_silences = agent_config.max_silence_retries
        self.max_clarifications = agent_config.max_clarification_retries

        # Initialize call data
        self.call_data = self._initialize_call_data()

        # Initialize components
        self.intent_detector = None
        self.parakeet_model = None
        self.crm_logger = None
        self.outcome_handler = None
        self.continuous_transcription = None  # Continuous listening handler
        self.silero_vad_singleton = None  # VAD singleton reference

        # Ringing detection (uuid_record-based)
        self.ringing_detected = False
        self.recording_file = None  # Detection recording (references full_recording_file)
        self.full_recording_file = None  # Full call recording path (entire call)
        self.detection_thread = None
        self.detection_stop_event = threading.Event()

        # Goertzel detection components
        self.goertzel = None
        self.cycle_tracker = None
        self.validator = None

        # Voicemail detection (VMD) components
        self.vmd_enabled = False
        self.vmd_classifier = None
        self.vmd_start_time = None
        self.vmd_complete = False
        self.voicemail_detected = False
        self.vmd_recording_buffer = bytearray()
        self.vmd_confidence = 0.0
        self.vmd_detection_duration = 7.0

        # Playback state tracking for continuous transcription
        self.is_playing_audio = False
        self.playback_start_time = None
        self.playback_periods = []  # Track (start_time, end_time) tuples for race condition fix

        # Chunked playback controller for pause/resume support
        # Initialized lazily on first use (after continuous_transcription is set up)
        self.chunked_playback_controller: Optional[ChunkedPlaybackController] = None

        # Current step context for pause intent routing
        self._current_step_for_pause: Optional[Dict] = None

        # Main response listening flag (continuous transcription runs when this is False)
        self.main_response_listening = False

        # Track if critical pitch was delivered (for NP vs DC disposition logic)
        self.pitch_delivered = False

        # Temp file tracking for cleanup (prevents file handle leaks)
        self.temp_files = []
        self.temp_files_lock = threading.Lock()

        # Debug artifact tracking
        self.debug_artifacts = []

        # Extract source IP from SIP headers (for transfer target)
        self.source_ip = info.getHeader("variable_sip_h_X-FS-Support") or \
                        info.getHeader("variable_sip_network_ip") or \
                        agent_config.server_ip

        # Initialize transfer manager (replicates pjsua2 ViciDialApiTransfer)
        # Create if at least one DID is configured (independent operation)
        self.transfer_manager = None
        if agent_config.did_transfer_qualified or agent_config.did_transfer_hangup:
            try:
                self.transfer_manager = FreeSWITCHTransferManager(
                    qualified_did=agent_config.did_transfer_qualified,
                    hangup_did=agent_config.did_transfer_hangup,
                    server_ip=self.source_ip,
                    logger=self.logger
                )
            except Exception as e:
                self._log_and_collect('error', f"Failed to initialize transfer manager: {e}")
                self.transfer_manager = None
        else:
            self._log_and_collect('info', "Transfer manager not initialized: no DIDs configured")

        # Transfer state tracking
        self._transfer_lock = threading.Lock()
        self._transfer_attempted = False

        self._log_and_collect('info', f"Handler created for {self.phone_number}")

    def _save_debug_artifacts(self, error_type: str, exception: Exception, file_path: str = None):
        """
        Save debug artifacts when EBADF or other file errors occur

        Args:
            error_type: Type of error (e.g., "EBADF_DETECTION", "EBADF_MAIN_RESPONSE", "VMD_PIPELINE")
            exception: The exception that occurred
            file_path: Optional file path that caused the error
        """
        try:
            import shutil
            from datetime import datetime

            # Create timestamped debug directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = f"/root/sip-bot/debug/{timestamp}_{error_type}_{self.uuid[:8]}"
            os.makedirs(debug_dir, exist_ok=True)

            self._log_and_collect('error', f"[DEBUG ARTIFACTS] Saving to: {debug_dir}")

            # Save error info
            error_info = f"""
=== DEBUG ARTIFACTS ===
Error Type: {error_type}
Call UUID: {self.uuid}
Timestamp: {timestamp}
Exception: {exception}
Exception Type: {type(exception).__name__}
File Path: {file_path}

=== File State ===
"""

            # Check file state if path provided
            if file_path:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    error_info += f"File exists: YES\n"
                    error_info += f"File size: {stat_info.st_size} bytes\n"
                    error_info += f"File permissions: {oct(stat_info.st_mode)}\n"
                    error_info += f"Modified time: {stat_info.st_mtime}\n"

                    # Try to copy the file
                    try:
                        dest_file = os.path.join(debug_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, dest_file)
                        error_info += f"File copied to: {dest_file}\n"
                    except Exception as e:
                        error_info += f"Failed to copy file: {e}\n"
                else:
                    error_info += f"File exists: NO\n"

            # Save error info to file
            with open(os.path.join(debug_dir, "error_info.txt"), 'w') as f:
                f.write(error_info)

            # Save full stack trace
            import traceback
            with open(os.path.join(debug_dir, "stack_trace.txt"), 'w') as f:
                f.write(traceback.format_exc())

            # Track debug dir for cleanup later
            self.debug_artifacts.append(debug_dir)

            self._log_and_collect('error', f"[DEBUG ARTIFACTS] ✅ Saved to {debug_dir}")

        except Exception as e:
            self._log_and_collect('error', f"[DEBUG ARTIFACTS] Failed to save artifacts: {e}", exc_info=True)

    def _initialize_call_data(self) -> Dict[str, Any]:
        """Initialize call data structure"""
        return {
            'id': None,
            'phone_number': self.phone_number,
            'caller_state': self.caller_state,
            'start_time': self.call_start_time,
            'end_time': None,
            'disposition': 'INITIATED',
            'transcript': '',
            'is_voicemail': False,
            'intent_detected': None,
            'uniqueid': f"voicebot_{self.uuid}_{int(self.call_start_time)}",
            'duration': 0,
            'call_result': 'UNKNOWN',
            'originating_agent': self.agent_config.agent_id,
            'e_agent_id': self.agent_config.agent_id,
            'campaign_id': self.agent_config.campaign_id,
            'vici_lead_id': self.vici_lead_id,
            'vici_list_id': self.vici_list_id,
            'vici_campaign_id': self.vici_campaign_id,
            'filename': None,
            'file_mime_type': None,
            'call_drop_step': None,
            'error': None,
            'transfer_target': None,
            'transfer_status': None,
            'transfer_response_code': None,
            'transfer_timestamp': None,
            'transfer_reason': None,
            'e_scripts_id': self.agent_config.script_id
        }

    def _log_and_collect(self, level: str, message: str, **kwargs):
        """Log message to both standard logger and call log collector"""
        # Log to standard logger (with optional exc_info and other kwargs)
        if level.lower() == 'info':
            self.logger.info(message, **kwargs)
        elif level.lower() == 'warning':
            self.logger.warning(message, **kwargs)
        elif level.lower() == 'error':
            self.logger.error(message, **kwargs)
        elif level.lower() == 'debug':
            self.logger.debug(message, **kwargs)

        # Also collect in call log collector for database storage
        if hasattr(self, 'call_log_collector'):
            self.call_log_collector.add_log(level.upper(), message)

    def _determine_call_outcome(self) -> CallOutcome:
        """
        Map disposition to CallOutcome enum for ViciDial reporting
        Based on pjsua2 fallback mapping (src/call_outcome_handler.py lines 230-241)

        Disposition codes from pjsua2:
        - A: Answering machine/voicemail
        - RI: Ringing (no answer)
        - DNC: Do Not Call
        - NI: Not Interested
        - HP: Hold/Press (person said "hold on" or "press 1")
        - CLBK: Callback requested
        - DAIR/DAIR 2: Did Answer Incomplete Response (silence)
        - DC: Disconnected (after progress)
        - NP: No Progress (early hangup)
        - SALE: Transferred/qualified
        """
        disposition = self.call_data.get('disposition', 'INITIATED')

        # Voicemail/Answering machine
        if disposition == 'A':
            return CallOutcome.VOICEMAIL

        # Ringing/No answer
        elif disposition == 'RI':
            return CallOutcome.RINGING

        # Negative intents (DNC, NI, HP, CLBK from pjsua2 lines 1368-1378)
        elif disposition in ['DNC', 'NI', 'HP', 'CLBK']:
            return CallOutcome.NEGATIVE_INTENT

        # Transferred/Sale
        elif disposition == 'SALE':
            return CallOutcome.TRANSFERRED

        # Not qualified (from pjsua2 lines 1733-1740, 1782-1784)
        elif disposition in ['DAIR', 'DAIR 2']:
            return CallOutcome.NOT_QUALIFIED

        # Disconnected after progress
        elif disposition == 'DC':
            return CallOutcome.HANGUP_SCRIPTED

        # No progress (early hangup - from fallback_map)
        elif disposition == 'NP':
            return CallOutcome.HANGUP_EARLY

        # Default: early hangup
        else:
            return CallOutcome.HANGUP_EARLY

    def _start_full_call_recording(self):
        """
        Start full call recording (stereo, entire call duration)
        This recording captures everything from call answer to hangup:
        - All delays, silences, pauses
        - Caller audio on LEFT channel
        - Bot TTS playback on RIGHT channel (auto-captured by FreeSWITCH)
        """
        try:
            # Create recording file path using call ID
            call_id = self.call_data.get('id') or self.uuid
            self.full_recording_file = f"/usr/local/freeswitch/recordings/{call_id}.wav"

            self._log_and_collect('info', f"[DEBUG RECORDING] Starting full call recording")
            self._log_and_collect('info', f"[DEBUG RECORDING] UUID: {self.uuid}")
            self._log_and_collect('info', f"[DEBUG RECORDING] File path: {self.full_recording_file}")

            # Enable stereo recording (LEFT=caller, RIGHT=bot)
            self.conn.execute("set", "RECORD_STEREO=true")
            self._log_and_collect('info', "[DEBUG RECORDING] Enabled STEREO recording for full call")

            # Set write buffer to 1 second chunks (16000 bytes at 8kHz 16-bit mono)
            self.conn.execute("set", "enable_file_write_buffering=16000")
            self._log_and_collect('info', "[DEBUG RECORDING] Set write buffering to 1 second chunks")

            # Start FreeSWITCH recording (will run for entire call)
            cmd = f"uuid_record {self.uuid} start {self.full_recording_file}"
            self._log_and_collect('info', f"[DEBUG RECORDING] Sending command: {cmd}")

            result = self.conn.api(cmd)

            if result:
                response = result.getBody()
                self._log_and_collect('info', f"[DEBUG RECORDING] FreeSWITCH response: '{response}'")

                # Parse response for success/error
                if response.startswith("-ERR"):
                    self._log_and_collect('error', f"❌ uuid_record FAILED: {response}")
                    self._log_and_collect('error', f"[DEBUG RECORDING] Command: {cmd}")
                    return False
                elif response.startswith("+OK") or "Success" in response:
                    self._log_and_collect('info', f"✅ uuid_record command accepted by FreeSWITCH")
                else:
                    self._log_and_collect('warning', f"⚠️ uuid_record unexpected response: '{response}'")

                # Wait for file to be created with tight polling (reduced from 0.5s fixed sleep)
                for attempt in range(20):  # Up to 200ms total
                    if os.path.exists(self.full_recording_file):
                        file_size = os.path.getsize(self.full_recording_file)
                        self._log_and_collect('info', f"✅ RECORDING FILE CREATED: {self.full_recording_file} ({file_size} bytes) [attempt {attempt+1}]")
                        self._log_and_collect('info', "  LEFT channel = Caller audio (inbound RTP)")
                        self._log_and_collect('info', "  RIGHT channel = Bot audio (playback commands auto-captured)")
                        return True
                    time.sleep(0.01)  # 10ms per attempt

                # File not created after 200ms
                self._log_and_collect('error', f"❌ RECORDING FILE NOT CREATED: {self.full_recording_file}")
                self._log_and_collect('error', f"[DEBUG RECORDING] FreeSWITCH said: '{response}'")
                self._log_and_collect('error', f"[DEBUG RECORDING] But file does not exist after 200ms")
                return False
            else:
                self._log_and_collect('error', "❌ Failed to start full call recording - no result from FreeSWITCH")
                return False

        except Exception as e:
            self._log_and_collect('error', f"Error starting full call recording: {e}", exc_info=True)
            return False

    def _start_audio_detection(self):
        """Start unified audio detection (ringing + voicemail) using EXISTING full call recording"""
        from src.config import VMD_ENABLED, VMD_DETECTION_DURATION, VMD_CONFIDENCE_THRESHOLD, VMD_MODEL_PATH

        try:
            # Use the existing full call recording file (already started by _start_full_call_recording)
            if not hasattr(self, 'full_recording_file') or not self.full_recording_file:
                self._log_and_collect('error', "No full recording file - detection cannot start")
                return

            # Point recording_file to the full call recording (detection reads from it)
            self.recording_file = self.full_recording_file
            self._log_and_collect('info', f"Detection using full call recording: {self.recording_file}")
            self._log_and_collect('info', "  Recording already active - detection will analyze LEFT channel (caller audio)")

            # Initialize ringing detection components
            self.goertzel = GoertzelDetector(sample_rate=8000, chunk_size=1024)
            self.cycle_tracker = RingCycleTracker(required_rings=2)
            self.validator = DetectionValidator(
                relative_threshold=7.0,         # Signal-to-background ratio (higher = stricter)
                max_strength_threshold=150.0,   # Maximum strength before pattern validation required
                frequency_balance_ratio=6.0,    # 440Hz/480Hz balance ratio (lower = stricter balance)
                min_energy=8e4,                 # Minimum energy threshold (80,000 - filters low energy)
                required_consecutive=3          # Consecutive detections required (more reliable)
            )

            # Initialize voicemail detection components (NEW)
            if VMD_ENABLED:
                from src.voicemail_detector_core import VoicemailClassifier
                self.vmd_enabled = True
                self.vmd_classifier = VoicemailClassifier(
                    model_path=VMD_MODEL_PATH,
                    confidence_threshold=VMD_CONFIDENCE_THRESHOLD,
                    logger=self.logger
                )
                self.vmd_start_time = time.time()
                self.vmd_detection_duration = VMD_DETECTION_DURATION
                self.vmd_recording_buffer = bytearray()
                self._log_and_collect('info', 
                    f"VMD enabled (duration={VMD_DETECTION_DURATION}s, "
                    f"threshold={VMD_CONFIDENCE_THRESHOLD})"
                )

            # Start unified detection thread
            self.detection_thread = threading.Thread(
                target=self._unified_detection_loop,
                daemon=True
            )
            self.detection_thread.start()

            # Register detection thread for cleanup
            if self.detection_thread:
                self.resource_manager.register(
                    'detection_thread',
                    self.detection_thread,
                    cleanup_func=self._cleanup_detection_thread
                )

            detection_types = []
            if True:  # Ringing always enabled
                detection_types.append("ringing")
            if self.vmd_enabled:
                detection_types.append("voicemail")

            self._log_and_collect('info', f"✅ Audio detection started: {', '.join(detection_types)}")

        except Exception as e:
            self._log_and_collect('error', f"Failed to start audio detection: {e}", exc_info=True)

    def _unified_detection_loop(self):
        """
        Unified detection loop - monitors for BOTH ringback tones AND voicemail
        Runs in background thread, analyzes same recording file in parallel
        Uses inotify for efficient file monitoring (falls back to polling if unavailable)
        """
        from src.file_event_watcher import FileEventWatcher
        from src import config

        last_position = 44  # Skip WAV header
        detection_start_time = time.time()
        chunks_analyzed = 0
        recording_fd = None
        file_watcher = None

        try:
            # Wait for file to be created
            wait_count = 0
            while not os.path.exists(self.recording_file) and wait_count < 40:
                time.sleep(0.05)
                wait_count += 1

            if not os.path.exists(self.recording_file):
                self._log_and_collect('error', "Recording file not created")
                return

            self._log_and_collect('info', f"Analyzing recording: {self.recording_file}")

            # Create file event watcher for efficient monitoring
            try:
                file_watcher = FileEventWatcher(
                    self.recording_file,
                    logger=self.logger,
                    use_inotify=config.USE_INOTIFY,
                    fallback_interval=config.POLLING_FALLBACK_INTERVAL
                )
            except Exception as e:
                self._log_and_collect('warning', f"[DETECTION] Failed to create file watcher: {e}, using polling")
                file_watcher = None

            # Open file once and keep descriptor open (POSIX allows multiple readers safely)
            try:
                # Log file state before opening
                if os.path.exists(self.recording_file):
                    file_size = os.path.getsize(self.recording_file)
                    self._log_and_collect('info', f"[DETECTION] File exists: {self.recording_file} ({file_size} bytes)")
                else:
                    self._log_and_collect('error', f"[DETECTION] File does NOT exist: {self.recording_file}")
                    return

                recording_fd = open(self.recording_file, 'rb')
                fd_number = recording_fd.fileno()
                self._log_and_collect('info', f"[DETECTION] Opened FD {fd_number} for {self.recording_file}")
            except (OSError, IOError) as e:
                self._log_and_collect('error', f"[DETECTION] Failed to open recording file: {e}")
                self._save_debug_artifacts("DETECTION_OPEN_FAILED", e, self.recording_file)
                return

            while self.is_active and not self.detection_stop_event.is_set():
                try:
                    # Wait for file modification using inotify (or polling fallback)
                    if file_watcher:
                        has_changes = file_watcher.wait_for_modification(timeout_ms=config.INOTIFY_TIMEOUT_MS)
                    else:
                        # Fallback: sleep for polling interval
                        time.sleep(config.POLLING_FALLBACK_INTERVAL)
                        has_changes = True

                    # Check if file still exists (caller may have hung up)
                    if not os.path.exists(self.recording_file):
                        self._log_and_collect('warning', "[DETECTION] Recording file disappeared (caller hung up)")
                        break

                    current_size = os.path.getsize(self.recording_file)

                    if current_size > last_position:
                        # Read new audio data using persistent file descriptor
                        try:
                            recording_fd.seek(last_position)
                            new_audio_stereo = recording_fd.read(current_size - last_position)
                        except (OSError, IOError) as e:
                            import errno
                            if e.errno == errno.EBADF:
                                self._log_and_collect('error', f"[DETECTION] ❌ EBADF ERROR: Bad file descriptor")
                                self._log_and_collect('error', f"[DETECTION] FD was: {recording_fd.fileno() if recording_fd else 'None'}")
                                self._log_and_collect('error', f"[DETECTION] File: {self.recording_file}")
                                self._log_and_collect('error', f"[DETECTION] Position: {last_position}")
                                self._log_and_collect('error', f"[DETECTION] Current size: {current_size}")
                                self._save_debug_artifacts("EBADF_DETECTION", e, self.recording_file)
                            else:
                                self._log_and_collect('warning', f"[DETECTION] Error reading from file: {e}")
                            break

                        if len(new_audio_stereo) > 0:
                            # Extract LEFT channel only (caller audio, not bot audio)
                            new_audio = self._extract_left_channel_from_stereo(new_audio_stereo)

                            # === PARALLEL DETECTION 1: RINGING (Left channel only) ===
                            self._detect_ringing(new_audio, chunks_analyzed)
                            if self.ringing_detected:
                                return  # Exit if ringing found

                            # === PARALLEL DETECTION 2: VOICEMAIL (Left channel only) ===
                            if self.vmd_enabled and not self.vmd_complete:
                                self._detect_voicemail(new_audio)
                                if self.voicemail_detected:
                                    return  # Exit if voicemail found

                            # === PARALLEL DETECTION 3: CONTINUOUS TRANSCRIPTION (always-on except during main response) ===
                            # Continuous transcription runs 95% of call time (only paused during main response recording)
                            if self.continuous_transcription:
                                # Check if main response listener is active
                                is_continuous_active = not self.main_response_listening

                                # Diagnostic logging (only log every 20 chunks to avoid spam)
                                if chunks_analyzed % 20 == 0:
                                    if is_continuous_active:
                                        self._log_and_collect('debug', f"[CONTINUOUS TRANSCRIPTION] Active (main_response_listening={self.main_response_listening}) - processing audio")
                                    else:
                                        self._log_and_collect('debug', f"[CONTINUOUS TRANSCRIPTION] Paused during main response recording")

                                if is_continuous_active:
                                    # Add audio chunk to transcription buffer (also updates VAD state)
                                    self.continuous_transcription.add_audio_chunk(new_audio)

                                    # Check for speech-silence patterns and transcribe when appropriate
                                    # The handler now uses Silero VAD and speech-silence detection
                                    # to decide when to transcribe (not fixed intervals)
                                    self.continuous_transcription.transcribe_and_check_intents()

                            chunks_analyzed += 1
                            last_position = current_size

                except Exception as e:
                    self._log_and_collect('error', f"Error in detection loop: {e}", exc_info=True)
                    # Fallback sleep on error
                    time.sleep(config.POLLING_FALLBACK_INTERVAL if hasattr(config, 'POLLING_FALLBACK_INTERVAL') else 0.5)

        except Exception as e:
            self._log_and_collect('error', f"Fatal error in unified detection: {e}", exc_info=True)
        finally:
            # Close file watcher
            if file_watcher:
                try:
                    file_watcher.close()
                    self._log_and_collect('debug', "[DETECTION] Closed file watcher")
                except Exception as e:
                    self._log_and_collect('debug', f"[DETECTION] Error closing file watcher: {e}")

            # Close persistent file descriptor
            if recording_fd:
                try:
                    recording_fd.close()
                    self._log_and_collect('debug', "[DETECTION] Closed persistent file descriptor")
                except Exception as e:
                    self._log_and_collect('debug', f"[DETECTION] Error closing file descriptor: {e}")

            detection_summary = []
            if self.ringing_detected:
                detection_summary.append("ringing=YES")
            if self.voicemail_detected:
                detection_summary.append(f"voicemail=YES (conf={self.vmd_confidence:.2f})")
            if not self.ringing_detected and not self.voicemail_detected:
                detection_summary.append("no detection")

            self._log_and_collect('info', 
                f"Unified detection stopped. Analyzed {chunks_analyzed} chunks. "
                f"Results: {', '.join(detection_summary)}"
            )

    def _detect_ringing(self, audio_bytes: bytes, chunk_num: int):
        """
        Detect ringback tones (440Hz/480Hz) - continuous analysis
        Extracted from original ringing detection loop for parallel execution
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Analyze in 1024-sample chunks
            chunk_size = 1024
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i+chunk_size]

                if len(chunk) < chunk_size // 2:
                    continue

                # Detect frequencies (440Hz, 480Hz for US ringback)
                energy_440 = self.goertzel.detect_frequency(chunk, 440.0)
                energy_480 = self.goertzel.detect_frequency(chunk, 480.0)

                # Calculate background energy
                bg_energies = [
                    self.goertzel.detect_frequency(chunk, freq)
                    for freq in [300, 350, 400, 500, 550, 600]
                ]
                avg_bg = np.mean(bg_energies) + 1e-9

                # Relative strength
                min_target = min(energy_440, energy_480)
                relative_strength = min_target / avg_bg

                # Validate detection
                is_ringing = self.validator.validate(
                    energy_440, energy_480, relative_strength, chunk
                )

                # Track ring cycles
                current_time = time.time()
                if self.cycle_tracker.update(is_ringing, current_time):
                    # Ringing detected!
                    self._handle_ringback_detected()
                    return

        except Exception as e:
            self._log_and_collect('error', f"Error in ringing detection: {e}")

    def _detect_voicemail(self, audio_bytes: bytes):
        """
        Detect voicemail - accumulate audio for duration, then classify with ML
        Only runs during the detection period (first 7 seconds)
        """
        from src.config import VMD_END_CALL_ON_DETECTION

        try:
            # === DIAGNOSTIC: Track audio chunk size ===
            chunk_size = len(audio_bytes)
            buffer_before = len(self.vmd_recording_buffer)

            # Accumulate audio for VMD analysis
            self.vmd_recording_buffer.extend(audio_bytes)

            buffer_after = len(self.vmd_recording_buffer)

            # === DIAGNOSTIC: Verify buffer growth ===
            if buffer_after != buffer_before + chunk_size:
                self._log_and_collect('warning', 
                    f"[VMD DIAGNOSTIC] Buffer growth mismatch: "
                    f"expected {buffer_before + chunk_size}, got {buffer_after}"
                )

            # Check if detection period is complete
            elapsed = time.time() - self.vmd_start_time

            # === DIAGNOSTIC: Calculate expected buffer size for 7s at 8kHz 16-bit ===
            expected_buffer_size = int(elapsed * 8000 * 2)  # 8kHz, 16-bit (2 bytes)
            buffer_ratio = buffer_after / expected_buffer_size if expected_buffer_size > 0 else 0

            if elapsed >= self.vmd_detection_duration:
                if not self.vmd_complete:
                    self.vmd_complete = True

                    # === DIAGNOSTIC: Detailed completion logging ===
                    expected_7s_buffer_mono = int(7.0 * 8000 * 2)  # 112,000 bytes (mono)
                    expected_7s_buffer_stereo = expected_7s_buffer_mono * 2  # 224,000 bytes (stereo)

                    # Get stereo file size for comparison
                    try:
                        stereo_file_size = os.path.getsize(self.recording_file) if os.path.exists(self.recording_file) else 0
                    except:
                        stereo_file_size = 0

                    self._log_and_collect('info', 
                        f"[VMD DIAGNOSTIC] Detection period complete:\n"
                        f"  Elapsed time: {elapsed:.3f}s (target: {self.vmd_detection_duration}s)\n"
                        f"  Stereo file size: {stereo_file_size:,} bytes\n"
                        f"  Mono buffer size (LEFT channel): {buffer_after:,} bytes\n"
                        f"  Expected mono: {expected_7s_buffer_mono:,} bytes (7s at 8kHz 16-bit)\n"
                        f"  Buffer ratio: {buffer_ratio:.2f}x\n"
                        f"  Audio duration: {buffer_after / (8000 * 2):.2f}s\n"
                        f"  Note: Analyzing LEFT channel only (caller audio, not bot)"
                    )

                    # Run ML classification
                    is_voicemail, confidence = self._classify_voicemail()

                    if is_voicemail:
                        self.voicemail_detected = True
                        self.vmd_confidence = confidence
                        self._log_and_collect('warning', 
                            f"🤖 VOICEMAIL DETECTED (confidence: {confidence:.2f})"
                        )

                        # Handle detection
                        if VMD_END_CALL_ON_DETECTION:
                            self._handle_voicemail_detected()
                    else:
                        self._log_and_collect('info', 
                            f"👤 LIVE PERSON detected (confidence: {confidence:.2f})"
                        )
            else:
                # === DIAGNOSTIC: Enhanced progress logging (every 1 second) ===
                if int(elapsed * 10) % 10 == 0 and int(elapsed * 10) != int((elapsed - 0.05) * 10):
                    remaining = self.vmd_detection_duration - elapsed
                    audio_duration = buffer_after / (8000 * 2)
                    self._log_and_collect('info', 
                        f"[VMD] Recording: {elapsed:.1f}s/{self.vmd_detection_duration}s "
                        f"| Buffer: {buffer_after:,} bytes ({audio_duration:.1f}s audio) "
                        f"| Ratio: {buffer_ratio:.2f}x"
                    )

        except Exception as e:
            self._log_and_collect('error', f"[VMD DIAGNOSTIC] Error in VMD detection: {e}", exc_info=True)

    def _classify_voicemail(self) -> tuple:
        """
        Classify accumulated audio as voicemail or live person
        Returns: (is_voicemail: bool, confidence: float)
        """
        from src.config import VMD_MIN_AUDIO_LENGTH, VMD_CONFIDENCE_THRESHOLD
        import numpy as np

        try:
            # === DIAGNOSTIC: Pre-classification validation ===
            buffer_size = len(self.vmd_recording_buffer)
            audio_duration = buffer_size / (8000 * 2)  # 8kHz, 16-bit

            self._log_and_collect('info', 
                f"[VMD DIAGNOSTIC] Pre-classification validation:\n"
                f"  Buffer size: {buffer_size:,} bytes\n"
                f"  Audio duration: {audio_duration:.2f}s\n"
                f"  Minimum required: {VMD_MIN_AUDIO_LENGTH:,} bytes\n"
                f"  Confidence threshold: {VMD_CONFIDENCE_THRESHOLD}"
            )

            # Check minimum audio length
            if buffer_size < VMD_MIN_AUDIO_LENGTH:
                self._log_and_collect('warning', 
                    f"[VMD DIAGNOSTIC] ❌ Insufficient audio for VMD "
                    f"({buffer_size:,} < {VMD_MIN_AUDIO_LENGTH:,} bytes)"
                )
                return False, 0.0

            # === DIAGNOSTIC: Audio quality validation ===
            try:
                audio_array = np.frombuffer(self.vmd_recording_buffer, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(float)**2))
                peak = np.max(np.abs(audio_array))

                self._log_and_collect('info', 
                    f"[VMD DIAGNOSTIC] Audio quality metrics:\n"
                    f"  Samples: {len(audio_array):,}\n"
                    f"  RMS energy: {rms:.1f}\n"
                    f"  Peak amplitude: {peak:.0f}\n"
                    f"  Duration check: {len(audio_array) / 8000:.2f}s"
                )

                # Warn about audio quality issues
                if rms < 100:
                    self._log_and_collect('warning', 
                        f"[VMD DIAGNOSTIC] ⚠️  Very low audio energy (RMS={rms:.1f}). "
                        f"Audio may be silent or nearly silent."
                    )
                if peak < 500:
                    self._log_and_collect('warning', 
                        f"[VMD DIAGNOSTIC] ⚠️  Very low peak amplitude (peak={peak:.0f}). "
                        f"Audio may be too quiet for reliable classification."
                    )

            except Exception as e:
                self._log_and_collect('warning', f"[VMD DIAGNOSTIC] Could not analyze audio quality: {e}")

            # === DIAGNOSTIC: Timing verification ===
            elapsed_since_start = time.time() - self.vmd_start_time
            self._log_and_collect('info', 
                f"[VMD DIAGNOSTIC] Classification timing:\n"
                f"  Time since VMD start: {elapsed_since_start:.3f}s\n"
                f"  Target duration: {self.vmd_detection_duration}s\n"
                f"  Classification triggered: {'ON TIME' if elapsed_since_start >= self.vmd_detection_duration else 'EARLY!'}"
            )

            if elapsed_since_start < self.vmd_detection_duration:
                self._log_and_collect('error', 
                    f"[VMD DIAGNOSTIC] 🚨 CRITICAL: Classification triggered EARLY! "
                    f"({elapsed_since_start:.2f}s < {self.vmd_detection_duration}s). "
                    f"This WILL cause false positives!"
                )

            # Convert to bytes
            audio_bytes = bytes(self.vmd_recording_buffer)

            self._log_and_collect('info', f"[VMD] Running ML classification on {len(audio_bytes):,} bytes...")

            # Use classifier (time the inference)
            classification_start = time.time()
            is_voicemail, confidence = self.vmd_classifier.classify_audio(audio_bytes)
            classification_time = time.time() - classification_start

            # === DIAGNOSTIC: Classification results ===
            self._log_and_collect('info', 
                f"[VMD DIAGNOSTIC] Classification complete:\n"
                f"  Result: {'VOICEMAIL' if is_voicemail else 'LIVE PERSON'}\n"
                f"  Confidence: {confidence:.3f}\n"
                f"  Threshold: {VMD_CONFIDENCE_THRESHOLD}\n"
                f"  Decision: {'ABOVE threshold' if confidence >= VMD_CONFIDENCE_THRESHOLD else 'BELOW threshold'}\n"
                f"  Inference time: {classification_time:.2f}s"
            )

            return is_voicemail, confidence

        except Exception as e:
            self._log_and_collect('error', f"[VMD DIAGNOSTIC] VMD classification error: {e}", exc_info=True)
            return False, 0.0

    def _handle_voicemail_detected(self):
        """Called when voicemail is detected - hangup immediately"""
        self._log_and_collect('warning', 
            f"🤖 VOICEMAIL DETECTED (confidence: {self.vmd_confidence:.2f}) - "
            f"Ending call immediately"
        )

        # Set disposition to A (Answering Machine) - only if not already set
        if self.call_data['disposition'] == 'INITIATED':
            self.call_data['disposition'] = 'A'
        else:
            self._log_and_collect('info', f"Voicemail detected but preserving disposition: {self.call_data['disposition']}")

        # Mark detection flag
        self.voicemail_detected = True

        # Stop detection
        self.detection_stop_event.set()

        # Signal main thread to stop
        self.is_active = False

        # Direct hangup from detection thread
        try:
            self.conn.execute("hangup", "NORMAL_CLEARING")
        except Exception as e:
            self._log_and_collect('error', f"Error hanging up: {e}")

    def _handle_ringback_detected(self):
        """Called when ringback is detected - hangup immediately"""
        self._log_and_collect('warning', "🔔 RINGBACK DETECTED - Ending call immediately")

        # Set disposition to RI (Ring, no answer) - only if not already set
        if self.call_data['disposition'] == 'INITIATED':
            self.call_data['disposition'] = 'RI'
        else:
            self._log_and_collect('info', f"Ringback detected but preserving disposition: {self.call_data['disposition']}")
        self.ringing_detected = True

        # Stop detection
        self.detection_stop_event.set()

        # Signal main thread to stop
        self.is_active = False

        # Direct hangup from detection thread
        try:
            self.conn.execute("hangup", "NORMAL_CLEARING")
        except Exception as e:
            self._log_and_collect('error', f"Error hanging up: {e}")

    def _extract_left_channel_from_stereo(self, stereo_bytes: bytes) -> bytes:
        """
        Extract left channel (caller audio) from stereo PCM bytes

        Stereo audio is interleaved: L R L R L R ... (left, right, left, right, ...)
        We extract only the left channel: L L L ... (caller's voice only)

        Args:
            stereo_bytes: Interleaved stereo PCM bytes (16-bit samples)

        Returns:
            Mono PCM bytes containing only left channel (caller audio)
        """
        try:
            import numpy as np

            # Convert to numpy array (16-bit signed integers)
            stereo_array = np.frombuffer(stereo_bytes, dtype=np.int16)

            # Check if we actually have stereo data (even number of samples)
            if len(stereo_array) % 2 != 0:
                self._log_and_collect('warning', f"Odd number of samples ({len(stereo_array)}), audio may not be stereo")
                # Return as-is, might be mono already
                return stereo_bytes

            # Extract left channel (every other sample starting at index 0)
            # Stereo: [L0, R0, L1, R1, L2, R2, ...]
            # Left:   [L0, L1, L2, ...]
            left_channel = stereo_array[0::2]

            return left_channel.tobytes()

        except Exception as e:
            self._log_and_collect('error', f"Error extracting left channel: {e}")
            # Fallback: return original bytes (better than crashing)
            return stereo_bytes

    def _stop_audio_detection(self):
        """Stop unified audio detection (ringing + voicemail) and cleanup"""
        try:
            # Signal stop
            self.detection_stop_event.set()

            # Wait for thread
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)

            # DON'T stop recording - it needs to run for the full call
            # Recording will be stopped and saved in _finalize_call()
            if self.recording_file:
                if os.path.exists(self.recording_file):
                    file_size = os.path.getsize(self.recording_file)
                    self._log_and_collect('info', f"Full call recording still active: {self.recording_file} ({file_size} bytes so far)")
                    self._log_and_collect('info', "  Recording will continue until call end and be saved to /var/www/recordings/")

            # Cleanup VMD classifier (release model reference)
            if self.vmd_classifier:
                try:
                    self.vmd_classifier.cleanup()
                    self.vmd_classifier = None
                except Exception as e:
                    self._log_and_collect('error', f"Error cleaning up VMD classifier: {e}")

            detection_summary = []
            if self.ringing_detected:
                detection_summary.append("ringing detected")
            if self.voicemail_detected:
                detection_summary.append(f"voicemail detected (conf={self.vmd_confidence:.2f})")
            if not detection_summary:
                detection_summary.append("no detections")

            self._log_and_collect('info', f"Audio detection stopped: {', '.join(detection_summary)}")

        except Exception as e:
            self._log_and_collect('error', f"Error stopping audio detection: {e}")

    def run(self):
        """
        Main call lifecycle
        This is the entry point - blocks until call completes
        """
        try:
            self._log_and_collect('info', "="*60)
            self._log_and_collect('info', f"📞 Call Started: {self.phone_number}")
            self._log_and_collect('info', "="*60)

            # Subscribe to events for this call
            self.conn.sendRecv("myevents")
            self.conn.sendRecv("linger")  # Keep connection alive after hangup

            # Initialize components
            self._initialize_components()

            # Answer the call
            self._answer_call()

            # Start full call recording (must be before detection so detection can use it)
            self._start_full_call_recording()

            # Start unified audio detection (ringing + voicemail, uuid_record-based)
            self._start_audio_detection()

            # Load call flow
            self.call_flow = parse_call_flow_from_string(
                self.agent_config.script_content
            )

            if not self.call_flow:
                raise ValueError("Failed to load call flow")

            self._log_and_collect('info', f"Call flow loaded: {self.call_flow.get('name', 'Unknown')}")

            # Override max retries from call flow JSON if specified
            if 'max_clarification_retries' in self.call_flow:
                self.max_clarifications = self.call_flow['max_clarification_retries']
                self._log_and_collect('info', f"Max clarifications overridden from call flow: {self.max_clarifications}")

            if 'max_silence_retries' in self.call_flow:
                self.max_consecutive_silences = self.call_flow['max_silence_retries']
                self._log_and_collect('info', f"Max silences overridden from call flow: {self.max_consecutive_silences}")

            # Execute conversation
            self._execute_call_flow()

            # Finalize
            self._finalize_call()

        except Exception as e:
            self._log_and_collect('error', f"Call error: {e}", exc_info=True)
            self.call_data['error'] = str(e)
            self.call_data['disposition'] = 'NP'

        finally:
            self._cleanup()

        self._log_and_collect('info', f"✅ Call completed: {self.uuid[:8]}")

    def _initialize_components(self):
        """Initialize business logic components"""
        try:
            # Intent detector
            self.intent_detector = IntentDetector(
                hp_phrases=self.agent_config.honey_pot_sentences
            )

            # Parakeet model (singleton - accesses same instance preloaded by bot_server)
            parakeet = ParakeetModelSingleton()
            self.parakeet_model = parakeet.get_model(self.logger)
            if not self.parakeet_model:
                self._log_and_collect('warning', "Parakeet model not available")

            # Silero VAD singleton for regular response transcription filtering
            # (preloaded by bot_server at startup, zero per-call overhead)
            self.silero_vad_singleton = SileroVADSingleton()
            self._log_and_collect('info', "Silero VAD singleton ready for response transcription")

            # Qwen intent detector (singleton - accesses same instance preloaded by bot_server)
            if QwenModelSingleton:
                qwen = QwenModelSingleton.get_instance()
                self.qwen_detector = qwen.get_detector(self.logger)
                self._log_and_collect('info', "Qwen intent detector initialized")
            else:
                self.qwen_detector = None
                self._log_and_collect('warning', "Qwen singleton not available")

            # CRM logger
            self.crm_logger = SuiteCRMLogger(
                self.agent_config,
                self.logger,
                self
            )

            # ViciDial integration
            if all([self.agent_config.server_url,
                   self.agent_config.username,
                   self.agent_config.password]):
                api = ViciDialAPI(
                    server_url=self.agent_config.server_url,
                    api_user=self.agent_config.username,
                    api_pass=self.agent_config.password
                )
                self.vicidial_integration = VoicebotViciDialIntegration(
                    api, self.agent_config.vicidial_campaign_id
                )
                self._log_and_collect('info', "ViciDial integration initialized")
            else:
                self.vicidial_integration = None
                self._log_and_collect('info', "ViciDial integration disabled - missing config")

            # Outcome handler
            self.outcome_handler = CallOutcomeHandler(
                vicidial_integration=self.vicidial_integration,
                logger=self.logger
            )

            # Continuous transcription handler
            if self.parakeet_model and self.intent_detector:
                self.continuous_transcription = ContinuousTranscriptionHandler(
                    parakeet_model=self.parakeet_model,
                    intent_detector=self.intent_detector,
                    logger=self.logger,
                    rnnt_confidence_threshold=self.agent_config.rnnt_confidence_threshold,
                    energy_threshold=self.agent_config.energy_threshold,  # From database
                    immediate_hangup_callback=self._perform_immediate_hangup_from_detection  # Emergency hangup via ESL
                )
                self._log_and_collect('info', f"Continuous transcription handler initialized with immediate hangup (energy_threshold={self.agent_config.energy_threshold})")

                # Register continuous transcription for cleanup (prevents memory leak)
                self.resource_manager.register(
                    'continuous_transcription',
                    self.continuous_transcription,
                    cleanup_func=self._cleanup_continuous_transcription
                )
            else:
                self._log_and_collect('warning', "Continuous transcription disabled - missing parakeet_model or intent_detector")

            # Register conversation log and playback periods for cleanup
            self.resource_manager.register(
                'conversation_log',
                self.conversation_log,
                cleanup_func=self._cleanup_conversation_log
            )
            self.resource_manager.register(
                'playback_periods',
                self.playback_periods,
                cleanup_func=self._cleanup_playback_periods
            )
            self.resource_manager.register(
                'temp_files',
                self.temp_files,
                cleanup_func=self._cleanup_temp_files
            )

            # Log call start to CRM
            call_id = self.crm_logger.log_call_start(self.call_data)
            if call_id:
                self.call_data['id'] = call_id
                self._log_and_collect('info', f"Call logged to CRM: ID={call_id}")

            self._log_and_collect('info', "Components initialized")

        except Exception as e:
            self._log_and_collect('error', f"Component initialization failed: {e}")
            raise

    def _is_channel_active(self) -> bool:
        """
        Check if channel still exists (caller hasn't hung up)
        Uses uuid_exists API call to verify channel status

        Returns:
            True if channel is active, False if disconnected
        """
        try:
            result = self.conn.api("uuid_exists", self.uuid)
            if result:
                response = result.getBody().strip()
                return response == "true"
            return False
        except Exception as e:
            self._log_and_collect('error', f"Error checking channel status: {e}")
            return False

    def _handle_premature_hangup(self):
        """
        Handle premature call drop with pitch awareness
        - NP (No Pitch): Disconnected before critical pitch was delivered
        - DC (Disconnected): Disconnected after critical pitch was delivered
        Preserves existing dispositions (DNC/NI/HP/A/RI)
        """
        if self.is_active:
            self._log_and_collect('warning', "⚠️ Call disconnected by caller")
            # Only set disposition if no explicit disposition yet (preserve DNC/NI/HP/A/RI)
            if self.call_data['disposition'] == 'INITIATED':
                if self.pitch_delivered:
                    self.call_data['disposition'] = 'DC'
                    self._log_and_collect('info', "Setting disposition: DC (disconnected after pitch delivered)")
                else:
                    self.call_data['disposition'] = 'NP'
                    self._log_and_collect('info', "Setting disposition: NP (no pitch delivered)")
            else:
                self._log_and_collect('info', f"Preserving existing disposition: {self.call_data['disposition']}")
            self.is_active = False

    def _answer_call(self):
        """Answer the call"""
        self._log_and_collect('info', "Answering call...")
        self.conn.execute("answer", "")
        time.sleep(0.2)  # Brief pause after answer
        self._log_and_collect('info', "Call answered")

    def _play_audio(self, audio_file: str, step: Optional[Dict] = None) -> bool:
        """
        Play audio file using simple blocking playback

        Args:
            audio_file: Audio file name
            step: Optional step dictionary with greetings/us_states flags

        Returns:
            True if playback succeeded, False if interrupted or failed
        """
        try:
            # Get full path to audio file
            audio_path = get_audio_path_for_agent(
                audio_file,
                self.agent_config.voice_location,
                greetings=False,  # Simplified - no time-based variants
                us_states=False   # Simplified - no state-based variants
            )

            if not audio_path or not os.path.exists(audio_path):
                self._log_and_collect('error', f"Audio file not found: {audio_path}")
                return False

            # Check if channel still exists (caller hasn't hung up)
            if not self._is_channel_active():
                self._handle_premature_hangup()
                return False

            # Check if call is still active before playing
            if not self.is_active:
                self._log_and_collect('warning', "Call not active - skipping playback")
                return False

            self._log_and_collect('info', f"Playing: {os.path.basename(audio_path)}")
            self.conversation_log.append(f"Bot: {os.path.basename(audio_path)}")

            # Mark playback start for continuous transcription
            self.is_playing_audio = True
            self.playback_start_time = time.time()
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_start()

            # Simple blocking playback
            self.conn.execute("playback", audio_path)

            # Mark playback end
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()

            self._log_and_collect('debug', f"Playback completed")

            # Check for DNC/NI/HP AFTER playback completes
            if self.continuous_transcription:
                detected, intent_type = self.continuous_transcription.has_dnc_ni_detection()
                if detected:
                    self._log_and_collect('warning', f"🚫 {intent_type} detected after playback - ending call")

                    # Map intent type to disposition
                    intent_map = {
                        "DNC": ("DNC", "do_not_call", "dnc_during_playback"),
                        "NI": ("NI", "not_interested", "ni_during_playback"),
                        "HP": ("HP", "hold_press", "hp_during_playback")
                    }

                    if intent_type in intent_map:
                        disposition, intent_detected, call_result = intent_map[intent_type]
                        # Only set if disposition not already set (preserve earlier detections)
                        if self.call_data['disposition'] == 'INITIATED':
                            self.call_data['disposition'] = disposition
                            self.call_data['intent_detected'] = intent_detected
                            self.call_data['call_result'] = call_result
                            self._log_and_collect('info', f"Setting disposition: {disposition} (detected during playback)")
                        else:
                            self._log_and_collect('info', f"{intent_type} detected but preserving disposition: {self.call_data['disposition']}")
                        self.is_active = False
                        return False

            return True

        except Exception as e:
            self._log_and_collect('error', f"Playback error: {e}", exc_info=True)
            # Make sure to clean up playback state
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()
            return False

    def _get_chunked_playback_controller(self) -> ChunkedPlaybackController:
        """
        Get or create the chunked playback controller.
        Lazy initialization to ensure continuous_transcription is ready.

        Auto pause/resume behavior:
        - When user speaks during playback, pause at next chunk boundary
        - After 5 seconds of silence, automatically resume playback
        """
        if self.chunked_playback_controller is None:
            self.chunked_playback_controller = ChunkedPlaybackController(
                conn=self.conn,
                uuid=self.uuid,
                continuous_transcription=self.continuous_transcription,
                logger=self.logger,
                chunk_duration_ms=500,  # 500ms chunks = max pause latency
                on_chunk_complete=self._on_chunk_complete,
                auto_resume_on_silence=True,  # Enable auto-resume after user stops speaking
                silence_threshold=5.0,  # Legacy: keep for backward compat
                pause_silence_threshold=config.RESPONSE_SILENCE_TIMEOUT,  # 0.7s silence to trigger intent check
                on_pause_intent_check=self._check_pause_intent  # Intent check callback during pause
            )
        return self.chunked_playback_controller

    def _on_chunk_complete(self, chunk_index: int, total_chunks: int):
        """Callback after each chunk completes playback"""
        # Optional: Log progress for debugging
        if chunk_index % 10 == 0:  # Log every 10 chunks (~5 seconds)
            self._log_and_collect('debug', f"[CHUNKED] Progress: chunk {chunk_index + 1}/{total_chunks}")

    def _check_pause_intent(self, transcription: str) -> tuple:
        """
        Check intent from transcription captured during playback pause.
        NOTE: Transcription is ALREADY done by CT during playback - no re-transcription needed!

        Args:
            transcription: Already-transcribed text from CT handler

        Returns:
            Tuple of (intent_type, transcription_text)
            intent_type: 'neutral', 'positive', 'negative', 'clarifying',
                         'do_not_call', 'not_interested', 'hold_press', or None
        """
        if not transcription or len(transcription.strip()) == 0:
            self._log_and_collect('debug', "[PAUSE INTENT] Empty transcription - treating as neutral")
            return "neutral", None

        text = transcription.strip()
        self._log_and_collect('info', f"[PAUSE INTENT] Checking intent for: '{text}'")

        # Pre-screen with keyword detector for DNC/NI/HP
        keyword_result = self.intent_detector.detect_intent(text)
        if keyword_result:
            intent, kw_confidence = keyword_result
            if intent in ["do_not_call", "not_interested", "hold_press"]:
                self._log_and_collect('info', f"[PAUSE INTENT] Keyword detected: {intent}")
                return intent, text

        # Use Qwen for nuanced classification
        if hasattr(self, 'qwen_detector') and self.qwen_detector and hasattr(self, '_current_step_for_pause'):
            try:
                question = self._extract_question_from_step(self._current_step_for_pause)
                qwen_result = self.qwen_detector.detect_intent(question, text, timeout=config.QWEN_TOTAL_TIMEOUT)

                if qwen_result:
                    self._log_and_collect('info', f"[PAUSE INTENT] Qwen classified: {qwen_result}")
                    # Return as-is - rebuttals have their own routing in call flow
                    return qwen_result, text
            except Exception as e:
                self._log_and_collect('warning', f"[PAUSE INTENT] Qwen failed: {e}")

        # Default to neutral if no clear intent
        return "neutral", text

    def _play_audio_chunked(self, audio_file: str, step: Optional[Dict] = None) -> bool:
        """
        Play audio file using chunked playback with pause/resume support.

        This method splits audio into small chunks (500ms) and plays them
        sequentially, allowing for pause/resume at chunk boundaries.

        Args:
            audio_file: Audio file name
            step: Optional step dictionary with greetings/us_states flags

        Returns:
            True if playback succeeded/completed, False if interrupted or failed
        """
        try:
            # Get full path to audio file
            audio_path = get_audio_path_for_agent(
                audio_file,
                self.agent_config.voice_location,
                greetings=False,
                us_states=False
            )

            if not audio_path or not os.path.exists(audio_path):
                self._log_and_collect('error', f"Audio file not found: {audio_path}")
                return False

            # Check if channel still exists (caller hasn't hung up)
            if not self._is_channel_active():
                self._handle_premature_hangup()
                return False

            # Check if call is still active before playing
            if not self.is_active:
                self._log_and_collect('warning', "Call not active - skipping playback")
                return False

            self._log_and_collect('info', f"Playing (chunked): {os.path.basename(audio_path)}")
            self.conversation_log.append(f"Bot: {os.path.basename(audio_path)}")

            # Mark playback start for continuous transcription
            self.is_playing_audio = True
            self.playback_start_time = time.time()
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_start()

            # Get or create chunked playback controller
            controller = self._get_chunked_playback_controller()

            # Play audio using chunked playback
            result = controller.play_audio(audio_path)

            # Mark playback end
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()

            # Handle result
            if result == ChunkedPlaybackResult.COMPLETED:
                self._log_and_collect('debug', "Chunked playback completed")

            elif result == ChunkedPlaybackResult.INTERRUPTED:
                self._log_and_collect('info', f"Chunked playback interrupted (barge-in) at {controller.progress_percent:.1f}%")
                # Barge-in detected - check for DNC/NI/HP
                return self._handle_playback_interrupt()

            elif result == ChunkedPlaybackResult.PAUSED:
                self._log_and_collect('info', f"Chunked playback paused at {controller.progress_percent:.1f}%")
                # External pause requested - return True to continue call flow
                return True

            elif result == ChunkedPlaybackResult.CHANNEL_GONE:
                self._log_and_collect('warning', "Channel gone during chunked playback")
                self._handle_premature_hangup()
                return False

            elif result == ChunkedPlaybackResult.FAILED:
                self._log_and_collect('error', "Chunked playback failed")
                return False

            elif result == ChunkedPlaybackResult.INTENT_DETECTED:
                intent = controller.detected_intent
                transcription = controller.detected_transcription
                self._log_and_collect('info', f"Playback interrupted by intent: {intent} ('{transcription}')")

                # Handle based on intent type
                if intent == "do_not_call":
                    self._handle_dnc()
                    return False
                elif intent == "not_interested":
                    self._handle_not_interested()
                    return False
                elif intent == "hold_press":
                    self._handle_honeypot()
                    return False
                else:
                    # Route to appropriate call flow step and signal to exit current step
                    self._handle_pause_intent_routing(intent, transcription)
                    return "ROUTED"  # Signal: step changed, exit immediately

            # Check for DNC/NI/HP AFTER playback completes (same as _play_audio)
            if self.continuous_transcription:
                detected, intent_type = self.continuous_transcription.has_dnc_ni_detection()
                if detected:
                    self._log_and_collect('warning', f"[BLOCKED] {intent_type} detected after chunked playback - ending call")
                    return self._handle_intent_detection(intent_type)

            return True

        except Exception as e:
            self._log_and_collect('error', f"Chunked playback error: {e}", exc_info=True)
            # Clean up playback state
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()
            return False

    def _handle_playback_interrupt(self) -> bool:
        """
        Handle playback interruption (barge-in detected).

        Returns:
            False to signal call should handle the interruption
        """
        if self.continuous_transcription:
            detected, intent_type = self.continuous_transcription.has_dnc_ni_detection()
            if detected:
                self._log_and_collect('warning', f"[BLOCKED] {intent_type} detected during barge-in")
                return self._handle_intent_detection(intent_type)

        # Barge-in but no negative intent - could be a question or positive response
        self._log_and_collect('info', "Barge-in detected but no negative intent")
        return False  # Signal to handle the response

    def _handle_intent_detection(self, intent_type: str) -> bool:
        """
        Handle detected intent (DNC/NI/HP) during or after playback.

        Args:
            intent_type: Type of intent detected ("DNC", "NI", "HP")

        Returns:
            False to signal call should end
        """
        intent_map = {
            "DNC": ("DNC", "do_not_call", "dnc_during_playback"),
            "NI": ("NI", "not_interested", "ni_during_playback"),
            "HP": ("HP", "hold_press", "hp_during_playback")
        }

        if intent_type in intent_map:
            disposition, intent_detected, call_result = intent_map[intent_type]
            # Only set if disposition not already set (preserve earlier detections)
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = disposition
                self.call_data['intent_detected'] = intent_detected
                self.call_data['call_result'] = call_result
                self._log_and_collect('info', f"Setting disposition: {disposition} (detected during playback)")
            else:
                self._log_and_collect('info', f"{intent_type} detected but preserving disposition: {self.call_data['disposition']}")
            self.is_active = False

        return False

    def _handle_pause_intent_routing(self, intent: str, transcription: str) -> bool:
        """
        Route to appropriate call flow step based on intent detected during pause.

        Args:
            intent: The detected intent (positive, negative, clarifying, rebuttal_question_*, etc.)
            transcription: The user's transcribed speech

        Returns:
            True to continue call flow, False to end
        """
        if not hasattr(self, '_current_step_for_pause') or not self._current_step_for_pause:
            self._log_and_collect('warning', "[PAUSE ROUTING] No current step context - continuing")
            return True

        step = self._current_step_for_pause
        if transcription:
            self.conversation_log.append(f"User (during playback): {transcription}")

        # Use existing intent-to-step mapping
        next_step = self._map_qwen_intent_to_step(step, intent)

        if next_step:
            self._log_and_collect('info', f"[PAUSE ROUTING] Routing {intent} -> {next_step}")
            self.current_step = next_step
            return True
        else:
            # No explicit route - use no_match_next or continue
            no_match = step.get('no_match_next')
            if no_match:
                self._log_and_collect('info', f"[PAUSE ROUTING] No match for {intent} -> using no_match_next: {no_match}")
                self.current_step = no_match
            else:
                self._log_and_collect('info', f"[PAUSE ROUTING] No match for {intent} and no no_match_next - continuing with current flow")
            return True

    def pause_playback(self):
        """
        Request pause of chunked playback at next chunk boundary.
        Non-blocking - playback will pause after current chunk finishes.
        """
        if self.chunked_playback_controller and self.chunked_playback_controller.is_playing:
            self.chunked_playback_controller.pause()
            self._log_and_collect('info', "Pause requested for chunked playback")

    def resume_playback(self) -> bool:
        """
        Resume chunked playback from where it was paused.

        Returns:
            True if resumed successfully, False otherwise
        """
        if self.chunked_playback_controller and self.chunked_playback_controller.is_paused:
            self._log_and_collect('info', "Resuming chunked playback")

            # Mark playback start again
            self.is_playing_audio = True
            self.playback_start_time = time.time()
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_start()

            result = self.chunked_playback_controller.resume()

            # Mark playback end
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()

            return result == ChunkedPlaybackResult.COMPLETED

        self._log_and_collect('warning', "No paused playback to resume")
        return False

    def get_playback_progress(self) -> dict:
        """
        Get current playback progress information.

        Returns:
            Dict with progress info (percent, position_ms, total_ms, etc.)
        """
        if self.chunked_playback_controller:
            return {
                'is_playing': self.chunked_playback_controller.is_playing,
                'is_paused': self.chunked_playback_controller.is_paused,
                'progress_percent': self.chunked_playback_controller.progress_percent,
                'position_ms': self.chunked_playback_controller.current_position_ms,
                'total_ms': self.chunked_playback_controller.total_duration_ms,
                'current_chunk': self.chunked_playback_controller.paused_chunk_index,
                'total_chunks': self.chunked_playback_controller.total_chunks
            }
        return {
            'is_playing': self.is_playing_audio,
            'is_paused': False,
            'progress_percent': 0,
            'position_ms': 0,
            'total_ms': 0,
            'current_chunk': 0,
            'total_chunks': 0
        }

    def _listen_for_response(self, timeout: int = 10, step: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Record and transcribe user response using Silero VAD with three-timeout system

        THREE-TIMEOUT SYSTEM:
        1. silence_timeout (from timeout param): Pre-speech silence limit
        2. max_wait_time (from config): Hard recording duration limit
        3. post_speech_silence (from config): Silence after speech to stop

        Args:
            timeout: Pre-speech silence timeout in seconds (from call flow step)
            step: Current call flow step (used to detect critical pitch delivery)

        Returns:
            Transcribed text, "TIMEOUT_NO_RESPONSE" if silent, or None on error
        """
        response_start_time = time.time()
        accumulated_audio = bytearray()  # ALL audio for full call recording (FreeSWITCH uuid_record)
        speech_only_audio = bytearray()  # ONLY VAD-confirmed speech chunks for Parakeet transcription
        temp_path = None
        recording_fd = None
        keepalive_stop_event = None
        keepalive_thread = None
        file_watcher = None  # Initialize before try block to avoid UnboundLocalError in finally

        # Three timeout values
        silence_timeout = timeout  # Pre-speech silence (from call flow)
        max_wait_time = config.RESPONSE_MAX_WAIT_TIME  # Hard limit (15s)
        post_speech_silence = config.RESPONSE_SILENCE_TIMEOUT  # Post-speech (1s)

        # Speech/silence state tracking
        speech_detected = False
        speech_start_time = None
        last_speech_time = None
        silence_start_time = None

        # VAD-based speech accumulation tracking
        speech_chunk_count = 0  # Number of VAD-positive chunks
        speech_duration_accumulated = 0.0  # Actual speech duration from chunk sizes

        try:
            # Set main response listening flag (pauses continuous transcription)
            self.main_response_listening = True
            self._log_and_collect('debug', "[MAIN RESPONSE] Pausing continuous transcription during response recording")

            # Check if channel still exists (caller hasn't hung up)
            if not self._is_channel_active():
                self._handle_premature_hangup()
                return None

            # Check if call is still active
            if not self.is_active:
                self._log_and_collect('warning', "Call not active - skipping recording")
                return None

            # Check if recording file exists
            if not self.recording_file:
                self._log_and_collect('error', f"[MAIN RESPONSE] No recording_file attribute set")
                return None

            if not os.path.exists(self.recording_file):
                self._log_and_collect('error', f"[MAIN RESPONSE] Recording file does NOT exist: {self.recording_file}")
                return None

            # Log file state
            file_size = os.path.getsize(self.recording_file)
            self._log_and_collect('info', f"[MAIN RESPONSE] Recording file exists: {self.recording_file} ({file_size} bytes)")

            # Use Silero VAD singleton for speech detection (no instantiation needed)
            self._log_and_collect('debug', f"[MAIN RESPONSE] Using Silero VAD singleton (threshold={self.agent_config.energy_threshold})")

            # Track starting position in file (current end of file)
            starting_position = os.path.getsize(self.recording_file)
            last_position = starting_position
            self._log_and_collect('info', f"[MAIN RESPONSE] Starting position in file: {starting_position} bytes")

            # Create file event watcher for efficient monitoring
            from src.file_event_watcher import FileEventWatcher
            try:
                file_watcher = FileEventWatcher(
                    self.recording_file,
                    logger=self.logger,
                    use_inotify=config.USE_INOTIFY,
                    fallback_interval=config.POLLING_FALLBACK_INTERVAL
                )
            except Exception as e:
                self._log_and_collect('warning', f"[MAIN RESPONSE] Failed to create file watcher: {e}, using polling")
                file_watcher = None

            # Open file with retry logic (handles transient FreeSwitch flush delays)
            MAX_OPEN_RETRIES = 3
            RETRY_DELAY_MS = 50

            for attempt in range(MAX_OPEN_RETRIES):
                try:
                    recording_fd = open(self.recording_file, 'rb')
                    fd_number = recording_fd.fileno()
                    self._log_and_collect('info', f"[MAIN RESPONSE] Opened FD {fd_number} for {self.recording_file} (attempt {attempt+1})")
                    break  # Success!
                except (OSError, IOError) as e:
                    is_last_attempt = (attempt == MAX_OPEN_RETRIES - 1)

                    if not is_last_attempt:
                        # Safety check: Call still active?
                        if not self._is_channel_active():
                            self._log_and_collect('warning', f"[MAIN RESPONSE] Channel died during file open retry")
                            return None

                        # Safety check: File still exists?
                        if not os.path.exists(self.recording_file):
                            self._log_and_collect('error', f"[MAIN RESPONSE] File disappeared during retry")
                            return None

                        # Log and retry
                        self._log_and_collect('warning', f"[MAIN RESPONSE] File open failed (attempt {attempt+1}/{MAX_OPEN_RETRIES}): {e}, retrying in {RETRY_DELAY_MS}ms")
                        time.sleep(RETRY_DELAY_MS / 1000.0)
                    else:
                        # All retries exhausted - treat as silence
                        self._log_and_collect('error',
                            f"[MAIN RESPONSE] Transcription failed: File open error after {MAX_OPEN_RETRIES} attempts: {e} - treating as silence"
                        )
                        self._save_debug_artifacts("MAIN_RESPONSE_OPEN_FAILED", e, self.recording_file)
                        return None

            # If we get here without recording_fd, all retries failed
            if recording_fd is None:
                self._log_and_collect('error',
                    "[MAIN RESPONSE] Transcription failed: No file descriptor obtained after retries - treating as silence"
                )
                return None

            self._log_and_collect('info', f"[MAIN RESPONSE] Listening with timeouts: silence={silence_timeout}s, max={max_wait_time}s, post_speech={post_speech_silence}s")

            # Keep channel active with interruptible sleep (maintains RTP flow for uuid_record)
            # uuid_record media bug requires dialplan application execution (not API commands)
            # Use threading.Event for immediate interruption when recording completes
            keepalive_stop_event = threading.Event()

            def keep_channel_active():
                """Run sleep in 500ms chunks for quick stop response while maintaining RTP"""
                try:
                    interval_ms = 500  # 500ms intervals for faster stop event detection
                    total_ms = int(max_wait_time * 1000)
                    elapsed = 0

                    while elapsed < total_ms:
                        if keepalive_stop_event.is_set():
                            self._log_and_collect('debug', f"[MAIN RESPONSE] Keepalive interrupted after {elapsed/1000:.1f}s")
                            return

                        remaining = min(interval_ms, total_ms - elapsed)
                        self.conn.execute("sleep", str(remaining))
                        elapsed += remaining

                    self._log_and_collect('debug', f"[MAIN RESPONSE] Keepalive completed full {max_wait_time}s")
                except Exception as e:
                    self._log_and_collect('error', f"[MAIN RESPONSE] Keepalive error: {e}", exc_info=True)

            keepalive_thread = threading.Thread(target=keep_channel_active, daemon=True, name="MainResponseKeepalive")
            keepalive_thread.start()
            self._log_and_collect('debug', f"[MAIN RESPONSE] Started interruptible keepalive (500ms chunks, max {max_wait_time}s)")

            # Mark that pitch was delivered if this is the critical step
            # Set flag HERE (right before listening loop) - not earlier during initialization
            # This ensures NP is set for hangups during setup, DC for hangups during actual listening
            if step and step.get('criticalstep', False):
                self.pitch_delivered = True
                self._log_and_collect('info', "⚡ Critical pitch delivered - actively listening for response")

                # Check if no audio activity detected by this point - end call with DAIR
                has_audio_activity = self._check_audio_activity()
                if not has_audio_activity:
                    self._log_and_collect('warning',
                        "No audio activity detected by critical step - ending call with DAIR")
                    self.call_data['disposition'] = 'DAIR'
                    self.is_active = False
                    return False

            # Main recording loop with Silero VAD and inotify
            # Use 100ms timeout for responsiveness (need to detect user speech quickly)
            INOTIFY_TIMEOUT_MS = 100

            while True:
                elapsed = time.time() - response_start_time

                # STOP CONDITION 1: Max wait time reached (hard limit)
                if elapsed >= max_wait_time:
                    self._log_and_collect('info', f"[MAIN RESPONSE] Max wait time ({max_wait_time}s) reached")
                    break

                # STOP CONDITION 2: No speech detected and silence timeout exceeded
                if not speech_detected and elapsed >= silence_timeout:
                    self._log_and_collect('info', f"[MAIN RESPONSE] Silence timeout ({silence_timeout}s) - no speech detected")
                    return "TIMEOUT_NO_RESPONSE"

                try:
                    # Wait for file modification using inotify (or polling fallback)
                    if file_watcher:
                        has_changes = file_watcher.wait_for_modification(timeout_ms=INOTIFY_TIMEOUT_MS)
                    else:
                        # Fallback: sleep for polling interval
                        time.sleep(INOTIFY_TIMEOUT_MS / 1000.0)
                        has_changes = True

                    # Check if file still exists (caller may have hung up)
                    if not os.path.exists(self.recording_file):
                        self._log_and_collect('warning', "[MAIN RESPONSE] Recording file disappeared (caller hung up)")
                        break

                    current_size = os.path.getsize(self.recording_file)

                    if current_size > last_position:
                        # Read new audio data with retry logic (handles transient FD invalidation)
                        MAX_READ_RETRIES = 3
                        RETRY_DELAY_MS = 50
                        new_audio_stereo = None
                        read_successful = False

                        for read_attempt in range(MAX_READ_RETRIES):
                            try:
                                recording_fd.seek(last_position)
                                new_audio_stereo = recording_fd.read(current_size - last_position)
                                read_successful = True
                                break  # Success!

                            except (OSError, IOError) as e:
                                import errno
                                is_last_attempt = (read_attempt == MAX_READ_RETRIES - 1)

                                # Log detailed error info
                                if e.errno == errno.EBADF:
                                    self._log_and_collect('error', f"[MAIN RESPONSE] ❌ EBADF during read (attempt {read_attempt+1}/{MAX_READ_RETRIES})")
                                    self._log_and_collect('error', f"[MAIN RESPONSE] FD was: {recording_fd.fileno() if recording_fd else 'None'}")
                                    self._log_and_collect('error', f"[MAIN RESPONSE] File: {self.recording_file}")
                                    self._log_and_collect('error', f"[MAIN RESPONSE] Position: {last_position}")
                                    self._log_and_collect('error', f"[MAIN RESPONSE] Current size: {current_size}")
                                    self._log_and_collect('error', f"[MAIN RESPONSE] Elapsed: {elapsed:.2f}s")
                                else:
                                    self._log_and_collect('warning', f"[MAIN RESPONSE] Error reading from file (attempt {read_attempt+1}/{MAX_READ_RETRIES}): {e}")

                                if not is_last_attempt:
                                    # Try to reopen the file with a fresh FD
                                    try:
                                        self._log_and_collect('warning', f"[MAIN RESPONSE] Attempting to reopen file after read error")
                                        if recording_fd:
                                            try:
                                                recording_fd.close()
                                            except:
                                                pass  # Already invalid, ignore close error

                                        recording_fd = open(self.recording_file, 'rb')
                                        new_fd = recording_fd.fileno()
                                        self._log_and_collect('info', f"[MAIN RESPONSE] Reopened file with new FD {new_fd}")
                                        time.sleep(RETRY_DELAY_MS / 1000.0)
                                        continue  # Retry the read

                                    except Exception as reopen_error:
                                        self._log_and_collect('error', f"[MAIN RESPONSE] Failed to reopen file: {reopen_error}")
                                        # Fall through to check if last attempt

                                if is_last_attempt:
                                    # All retries exhausted
                                    self._save_debug_artifacts("EBADF_MAIN_RESPONSE", e, self.recording_file)

                                    # If speech was detected, treat as transcription failure (silence)
                                    # If no speech, this is just a read error during silence
                                    if speech_detected:
                                        self._log_and_collect('error',
                                            "[MAIN RESPONSE] Transcription failed: Bad file descriptor (EBADF) while reading recording - treating as silence"
                                        )
                                        return None
                                    else:
                                        self._log_and_collect('warning', "[MAIN RESPONSE] EBADF with no speech - breaking loop")
                                        break  # Exit read retry loop

                                break  # Exit read retry loop if we couldn't recover

                        # Skip processing if read failed
                        if not read_successful or not new_audio_stereo:
                            break  # Exit main recording loop

                        if len(new_audio_stereo) > 0:
                            # Extract LEFT channel only (caller audio, not bot audio)
                            new_audio = self._extract_left_channel_from_stereo(new_audio_stereo)

                            # Check for speech using Silero VAD singleton
                            is_speech = self.silero_vad_singleton.is_speech(
                                new_audio,
                                threshold=self.agent_config.energy_threshold,
                                sample_rate=8000
                            )

                            if is_speech:
                                # Speech detected in this chunk
                                if not speech_detected:
                                    speech_detected = True
                                    speech_start_time = time.time()
                                    self._log_and_collect('debug', f"[MAIN RESPONSE] Speech detected at {elapsed:.2f}s")

                                last_speech_time = time.time()
                                silence_start_time = None  # Reset silence tracking
                            else:
                                # No speech in this chunk
                                if speech_detected:
                                    # User was speaking, now silent
                                    if silence_start_time is None:
                                        silence_start_time = time.time()
                                        self._log_and_collect('debug', f"[MAIN RESPONSE] Silence started at {elapsed:.2f}s")

                                    silence_duration = time.time() - silence_start_time

                                    # STOP CONDITION 3: Post-speech silence threshold
                                    if silence_duration >= post_speech_silence:
                                        self._log_and_collect('info', f"[MAIN RESPONSE] {post_speech_silence}s silence after speech - stopping")
                                        break

                            # Accumulate ALL audio for full call recording (FreeSWITCH uuid_record)
                            accumulated_audio.extend(new_audio)

                            # Accumulate ONLY VAD-confirmed speech chunks for Parakeet transcription
                            if is_speech:
                                # Speech chunk - add to speech buffer
                                speech_only_audio.extend(new_audio)
                                speech_chunk_count += 1

                                # Track actual speech duration from chunk size
                                chunk_duration = len(new_audio) / (8000 * 2)  # 8kHz, 16-bit (2 bytes/sample)
                                speech_duration_accumulated += chunk_duration

                                self._log_and_collect('debug',
                                    f"[MAIN RESPONSE] Speech chunk #{speech_chunk_count}: {chunk_duration:.3f}s "
                                    f"(total speech: {speech_duration_accumulated:.2f}s)"
                                )

                            # Mark audio activity for DAIR 2 detection (same logic as continuous transcription)
                            if len(new_audio) > 0 and self.continuous_transcription and not self.continuous_transcription.audio_activity_detected:
                                import numpy as np
                                audio_array = np.frombuffer(new_audio, dtype=np.int16)
                                rms = np.sqrt(np.mean(audio_array.astype(float)**2))

                                if rms > 50:
                                    self._log_and_collect('info', f"[MAIN RESPONSE] Audio activity detected (RMS={rms:.1f}) - will use DAIR 2 disposition if max silences reached")
                                    self.continuous_transcription.audio_activity_detected = True

                            last_position = current_size

                except Exception as e:
                    self._log_and_collect('error', f"[MAIN RESPONSE] Error reading audio chunk: {e}", exc_info=True)
                    # Sleep briefly on error
                    time.sleep(INOTIFY_TIMEOUT_MS / 1000.0)

            # Validate speech detection and duration
            if not speech_detected:
                self._log_and_collect('info', f"[MAIN RESPONSE] No speech detected during recording")
                return "TIMEOUT_NO_RESPONSE"

            # Check minimum speech duration (using accumulated speech duration, not time range)
            if speech_duration_accumulated < config.RESPONSE_MIN_SPEECH_DURATION:
                time_range = last_speech_time - speech_start_time if (speech_start_time and last_speech_time) else 0.0
                self._log_and_collect('info', 
                    f"[MAIN RESPONSE] Speech too short: actual={speech_duration_accumulated:.2f}s < {config.RESPONSE_MIN_SPEECH_DURATION}s "
                    f"(time_range={time_range:.2f}s, chunks={speech_chunk_count})"
                )
                return "TIMEOUT_NO_RESPONSE"

            time_range = last_speech_time - speech_start_time if (speech_start_time and last_speech_time) else 0.0
            self._log_and_collect('debug', 
                f"[MAIN RESPONSE] Speech duration: actual={speech_duration_accumulated:.2f}s, "
                f"time_range={time_range:.2f}s, chunks={speech_chunk_count}"
            )

            # Check if we collected enough SPEECH audio (fail gracefully if VAD filtered everything)
            if len(speech_only_audio) < 1000:  # Less than 1KB
                self._log_and_collect('info', 
                    f"[MAIN RESPONSE] Insufficient speech audio collected: {len(speech_only_audio)} bytes "
                    f"(total audio: {len(accumulated_audio)} bytes, chunks: {speech_chunk_count})"
                )
                return "TIMEOUT_NO_RESPONSE"

            elapsed_total = time.time() - response_start_time
            self._log_and_collect('info', 
                f"[MAIN RESPONSE] Collected speech={len(speech_only_audio)} bytes ({speech_duration_accumulated:.2f}s), "
                f"total={len(accumulated_audio)} bytes over {elapsed_total:.2f}s"
            )

            # Save SPEECH-ONLY audio to a temporary file for transcription
            # (Parakeet transcriber expects a file path)
            # Retry logic handles intermittent EBADF errors under concurrent load
            import wave
            import errno

            MAX_WAV_RETRIES = 3
            temp_path = None

            for wav_attempt in range(MAX_WAV_RETRIES):
                temp_path = f"/tmp/main_response_{uuid.uuid4().hex[:10]}.wav"

                try:
                    self._log_and_collect('debug', f"[MAIN RESPONSE] Creating temp WAV file: {temp_path} (attempt {wav_attempt+1}/{MAX_WAV_RETRIES})")
                    self._log_and_collect('debug', 
                        f"[MAIN RESPONSE] Audio size: speech={len(speech_only_audio)} bytes, "
                        f"total={len(accumulated_audio)} bytes, chunks={speech_chunk_count}"
                    )

                    # Write SPEECH-ONLY audio as WAV file (mono, 8000 Hz, 16-bit PCM)
                    with wave.open(temp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(8000)  # 8kHz
                        wav_file.writeframes(bytes(speech_only_audio))

                    # Success! Add to cleanup list AFTER successful write (prevents race condition)
                    with self.temp_files_lock:
                        self.temp_files.append(temp_path)

                    self._log_and_collect('info', 
                        f"[MAIN RESPONSE] Saved SPEECH-ONLY audio to {temp_path} for transcription "
                        f"({len(speech_only_audio)} bytes speech from {len(accumulated_audio)} bytes total)"
                    )
                    break  # Success, exit retry loop

                except OSError as wav_error:
                    if wav_error.errno == errno.EBADF and wav_attempt < MAX_WAV_RETRIES - 1:
                        # EBADF - transient filesystem/kernel race condition under concurrent load
                        # Same issue as voicemail detector (voicemail_detector_core.py:415-416)
                        self._log_and_collect('warning', 
                            f"[MAIN RESPONSE] ⚠️ EBADF on WAV write (attempt {wav_attempt+1}/{MAX_WAV_RETRIES}), "
                            f"retrying after 20ms delay (concurrent load race condition)..."
                        )
                        time.sleep(0.02)  # 20ms delay for filesystem sync
                        continue
                    else:
                        # Final attempt failed or different error - log and re-raise
                        self._log_and_collect('error', f"[MAIN RESPONSE] ❌ WAV write failed: {wav_error}")
                        self._log_and_collect('error', f"[MAIN RESPONSE] Error number: {wav_error.errno}")
                        self._log_and_collect('error', f"[MAIN RESPONSE] Is EBADF: {wav_error.errno == errno.EBADF}")
                        self._log_and_collect('error', f"[MAIN RESPONSE] File path: {temp_path}")
                        self._log_and_collect('error', f"[MAIN RESPONSE] File exists: {os.path.exists(temp_path)}")
                        if os.path.exists(temp_path):
                            self._log_and_collect('error', f"[MAIN RESPONSE] File size: {os.path.getsize(temp_path)} bytes")
                        self._save_debug_artifacts("WAV_WRITE_EBADF", wav_error, temp_path)
                        raise  # Re-raise to be caught by outer exception handler
            else:
                # All retries exhausted without success (shouldn't reach here due to raise above)
                raise Exception(f"WAV write failed after {MAX_WAV_RETRIES} attempts")

            # Transcribe using Parakeet with retry logic
            # Speech was detected by Silero VAD, so transcription failure is treated as silence
            MAX_RETRIES = 2  # 1 initial + 2 retries = 3 total attempts
            attempt = 0
            last_confidence = 0.0  # Track last failure reason

            while attempt <= MAX_RETRIES:
                attempt += 1

                if attempt > 1:
                    self._log_and_collect('warning', f"[MAIN RESPONSE] Transcription retry {attempt-1}/{MAX_RETRIES}")

                text, confidence = self._transcribe_audio(temp_path)
                self._log_and_collect('debug', f"[MAIN RESPONSE] _transcribe_audio returned: text={'<empty>' if not text else repr(text[:50])}, confidence={confidence:.3f}, threshold={self.agent_config.rnnt_confidence_threshold}")
                last_confidence = confidence  # Track for final error message

                if text:
                    self._log_and_collect('info', f"Transcribed: '{text}' (confidence: {confidence:.2f}) [attempt {attempt}]")
                    self.conversation_log.append(f"User: {text}")
                    return text

                # Transcription failed, wait briefly before retry (if not last attempt)
                if attempt <= MAX_RETRIES:
                    self._log_and_collect('debug', f"[MAIN RESPONSE] Transcription attempt {attempt} failed, waiting 500ms before retry")
                    time.sleep(0.5)  # 500ms between retries

            # All transcription attempts failed - log specific reason and treat as silence
            if last_confidence > 0.0:
                # Low confidence was the issue
                self._log_and_collect('error',
                    f"[MAIN RESPONSE] All transcription attempts failed: Confidence below threshold "
                    f"(confidence={last_confidence:.2f}, threshold={self.agent_config.rnnt_confidence_threshold}) - treating as silence"
                )
            else:
                # Model unavailable or exception (see logs above for details)
                self._log_and_collect('error',
                    "[MAIN RESPONSE] All transcription attempts failed: Model unavailable or exception "
                    "(see previous logs for details) - treating as silence"
                )
            return None

        except Exception as e:
            self._log_and_collect('error', f"Recording error: {e}", exc_info=True)

            # If VAD detected speech, log details but treat as silence
            # The error occurred during file I/O or transcription processing, not during speech detection
            if speech_detected:
                # Use accumulated speech duration instead of time range
                time_range = last_speech_time - speech_start_time if (speech_start_time and last_speech_time) else 0
                self._log_and_collect('error',
                    f"[MAIN RESPONSE] Transcription failed: Exception during processing (speech detected: "
                    f"duration={speech_duration_accumulated:.2f}s, time_range={time_range:.2f}s, "
                    f"chunks={speech_chunk_count}) - treating as silence"
                )
                return None
            else:
                self._log_and_collect('warning', "[MAIN RESPONSE] No speech detected and error occurred - treating as silence")
                return None

        finally:
            # Close file watcher
            if file_watcher:
                try:
                    file_watcher.close()
                    self._log_and_collect('debug', "[MAIN RESPONSE] Closed file watcher")
                except Exception as e:
                    self._log_and_collect('debug', f"[MAIN RESPONSE] Error closing file watcher: {e}")

            # Close persistent file descriptor
            if recording_fd:
                try:
                    recording_fd.close()
                    self._log_and_collect('debug', "[MAIN RESPONSE] Closed persistent file descriptor")
                except Exception as e:
                    self._log_and_collect('debug', f"[MAIN RESPONSE] Error closing file descriptor: {e}")

            # Stop the keepalive thread immediately using Event
            # Signal thread to exit, then wait for clean shutdown
            if keepalive_stop_event is not None:
                try:
                    keepalive_stop_event.set()  # Signal thread to stop
                    self._log_and_collect('debug', "[MAIN RESPONSE] Signaled keepalive thread to stop")

                    # Wait for thread to exit (max 500ms sleep interval + 100ms buffer)
                    if keepalive_thread is not None and keepalive_thread.is_alive():
                        keepalive_thread.join(timeout=0.6)
                        if keepalive_thread.is_alive():
                            self._log_and_collect('warning', "[MAIN RESPONSE] Keepalive thread didn't exit (daemon will clean up)")
                        else:
                            self._log_and_collect('debug', "[MAIN RESPONSE] Keepalive thread exited cleanly")
                except Exception as e:
                    self._log_and_collect('debug', f"[MAIN RESPONSE] Error stopping keepalive thread: {e}")

            # Check if caller hung up during recording (now safe - keepalive stopped)
            # This runs after recording loop completes naturally (timeout/silence)
            # If channel is gone, set NP and exit cleanly
            if not self._is_channel_active():
                self._log_and_collect('warning', "[MAIN RESPONSE] Channel disconnected during recording - setting NP")
                self._handle_premature_hangup()
                # Note: Returning from finally overrides any previous return value
                # This ensures we return None instead of TIMEOUT_NO_RESPONSE or transcription
                return None

            # Clear main response listening flag (resumes continuous transcription)
            self.main_response_listening = False
            self._log_and_collect('debug', "[MAIN RESPONSE] Resuming continuous transcription after response recording")

            # Note: Temp file cleanup handled by ResourceManager in _cleanup_temp_files()
            # Tracked in self.temp_files list for batch cleanup with retry logic

    def _transcribe_audio(self, audio_path: str) -> tuple:
        """
        Transcribe audio using Parakeet
        SIMPLE APPROACH - No VAD pre-filter, No retry logic
        Proven pattern from deprecated pjsua2 code

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (text, confidence)
        """
        try:
            if not self.parakeet_model:
                self._log_and_collect('error', "Transcription failed: Parakeet model not available")
                return None, 0.0

            # Use Parakeet to transcribe (NO VAD PRE-FILTER)
            # Parakeet has its own internal VAD - trust it
            text, confidence = self.parakeet_model.transcribe_with_confidence(
                audio_path
            )

            # Check confidence threshold
            if confidence < self.agent_config.rnnt_confidence_threshold:
                self._log_and_collect('warning',
                    f"Transcription failed: Confidence below threshold (confidence={confidence:.2f}, threshold={self.agent_config.rnnt_confidence_threshold})"
                )
                return None, confidence

            return text, confidence

        except Exception as e:
            self._log_and_collect('error', f"Transcription failed: {e}")
            return None, 0.0

    def _check_audio_activity(self) -> bool:
        """
        Check if there was any audio activity during the call.
        Used to differentiate DAIR (complete silence/carrier issue) from DAIR 2 (audio but no speech).

        Returns:
            True if audio activity detected, False if complete silence
        """
        if self.continuous_transcription:
            return self.continuous_transcription.has_audio_activity()
        return False

    def _transfer_call(self, did: str, is_qualified: bool = True) -> bool:
        """
        Transfer call using FreeSWITCH uuid_deflect (SIP REFER).
        Replicates pjsua2 transfer functionality.

        Args:
            did: DID number to transfer to
            is_qualified: True for qualified transfer, False for hangup transfer

        Returns:
            True if transfer succeeded, False otherwise
        """
        # Check if transfer manager is available
        if not self.transfer_manager:
            self._log_and_collect('error', "Transfer manager not initialized - cannot transfer")
            self.call_data['transfer_status'] = 'FAILED'
            self.call_data['transfer_reason'] = 'Transfer manager not initialized'
            return False

        # Prevent duplicate transfers (thread-safe)
        with self._transfer_lock:
            if self._transfer_attempted:
                self._log_and_collect('warning', "Transfer already attempted - skipping duplicate")
                return False
            self._transfer_attempted = True

        try:
            self._log_and_collect('info', f"Transferring to DID {did} (qualified={is_qualified})")

            # Update call data
            self.call_data['transfer_target'] = did
            self.call_data['transfer_target_ip'] = self.source_ip
            self.call_data['transfer_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.call_data['transfer_type'] = 'qualified' if is_qualified else 'hangup'

            # Execute transfer using transfer manager
            if is_qualified:
                success = self.transfer_manager.transfer_qualified_call(
                    self.conn,
                    self.uuid,
                    self.phone_number
                )
            else:
                success = self.transfer_manager.hangup_call_via_did(
                    self.conn,
                    self.uuid
                )

            # Update call data based on result
            if success:
                self.call_data['transfer_status'] = 'SUCCESS'
                self._log_and_collect('info', f"✅ Transfer successful to {did}")

                # Only set disposition if not already set
                if is_qualified and self.call_data['disposition'] == 'INITIATED':
                    self.call_data['disposition'] = 'QUALIFIED'
                elif not is_qualified and self.call_data['disposition'] == 'INITIATED':
                    # Hangup transfer - preserve or set appropriate disposition
                    pass

                # Mark call as transferred
                self.is_active = False
                return True
            else:
                self.call_data['transfer_status'] = 'FAILED'
                self.call_data['transfer_reason'] = 'Transfer attempts exhausted'
                self._log_and_collect('error', f"❌ Transfer failed to {did}")
                return False

        except Exception as e:
            self._log_and_collect('error', f"Transfer exception: {e}", exc_info=True)
            self.call_data['transfer_status'] = 'FAILED'
            self.call_data['transfer_reason'] = str(e)
            return False

    def _wait_for_transfer_confirmation(self, timeout: float = 5.0) -> bool:
        """
        Wait for transfer confirmation events from FreeSWITCH.

        Listens for CHANNEL_BRIDGE or CHANNEL_UNBRIDGE events to confirm
        transfer success/failure.

        Args:
            timeout: Maximum time to wait for confirmation (seconds)

        Returns:
            True if transfer confirmed successful, False otherwise

        Note:
            This is an optional enhancement for event-driven transfer confirmation.
            The uuid_deflect API response already provides immediate feedback.
        """
        try:
            import time
            start_time = time.time()

            while time.time() - start_time < timeout:
                # Receive event (non-blocking with 100ms timeout)
                event = self.conn.recvEvent()

                if event:
                    event_name = event.getHeader("Event-Name")
                    event_uuid = event.getHeader("Unique-ID")

                    # Only process events for this call
                    if event_uuid != self.uuid:
                        continue

                    if event_name == "CHANNEL_BRIDGE":
                        # Bridge successful - transfer is working
                        bridge_to = event.getHeader("Other-Leg-Unique-ID")
                        self._log_and_collect('info', f"✅ Transfer confirmed: CHANNEL_BRIDGE to {bridge_to}")
                        return True

                    elif event_name == "CHANNEL_UNBRIDGE":
                        # Unbridge - could be transfer completion or failure
                        self._log_and_collect('info', "Transfer event: CHANNEL_UNBRIDGE")

                    elif event_name == "CHANNEL_HANGUP":
                        # Call ended - transfer may have completed
                        hangup_cause = event.getHeader("Hangup-Cause")
                        self._log_and_collect('info', f"Transfer ended: CHANNEL_HANGUP ({hangup_cause})")
                        return hangup_cause == "NORMAL_CLEARING"

                # Small delay to avoid busy-waiting
                time.sleep(0.1)

            self._log_and_collect('warning', f"Transfer confirmation timeout after {timeout}s")
            return False

        except Exception as e:
            self._log_and_collect('error', f"Error waiting for transfer confirmation: {e}", exc_info=True)
            return False

    def _hangup_call(self):
        """
        Hangup the call with optional transfer to hangup DID.
        Replicates pjsua2's _perform_hangup_safe functionality.
        """
        try:
            self._log_and_collect('info', "Hanging up call")

            # Try transfer-based hangup first (if transfer manager available)
            if self.transfer_manager:
                # Check if not already transferred
                if not self._transfer_attempted:
                    # Transfer manager will check internally if hangup_did is configured
                    hangup_did = self.agent_config.did_transfer_hangup or self.transfer_manager.hangup_did
                    # Explicit check: Only attempt transfer if hangup_did is configured AND not None
                    if hangup_did and hangup_did != 'None' and str(hangup_did).strip():
                        self._log_and_collect('info', f"Attempting hangup via transfer to {hangup_did}")
                        if self._transfer_call(hangup_did, is_qualified=False):
                            self._log_and_collect('info', "✅ Hangup transfer successful")
                            self.call_data['call_result'] = 'transferred_hangup'

                            # Optional: Wait for event confirmation
                            # if self._wait_for_transfer_confirmation(timeout=2.0):
                            #     self._log_and_collect('info', "Transfer confirmed via events")

                            # Transfer will end the call, so just mark inactive
                            self.is_active = False
                            return
                    else:
                        self._log_and_collect('info', "Hangup transfer skipped: no hangup_did configured")

            # Fallback to direct hangup (original simple behavior)
            self._log_and_collect('info', "Using direct hangup (no transfer)")
            self.conn.execute("hangup", "")
            self.is_active = False

        except Exception as e:
            self._log_and_collect('error', f"Hangup error: {e}", exc_info=True)
            self.is_active = False

    def _perform_hangup_immediate(self, disposition: str, intent_type: str, call_result: str):
        """
        Flag for immediate hangup with disposition update
        Main thread handles actual hangup to avoid blocking

        Args:
            disposition: Call disposition (DNC, NI, HP)
            intent_type: Intent type for logging (do_not_call, not_interested, hold_press)
            call_result: Call result string for logging (e.g., 'dnc_during_playback')
        """
        self._log_and_collect('warning', f"🚫 Flagging immediate hangup: {disposition} detected")

        # Signal main thread to stop (no blocking execute() call!)
        self.is_active = False

        # Update call data - only if disposition not already set
        if self.call_data['disposition'] == 'INITIATED':
            self.call_data['disposition'] = disposition
            self.call_data['intent_detected'] = intent_type
            self.call_data['call_result'] = call_result
        else:
            self._log_and_collect('info', f"Immediate hangup ({disposition}) but preserving disposition: {self.call_data['disposition']}")

        # NOTE: Main thread will handle actual hangup in cleanup
        # Transcriptions collected in _finalize_call() (finally block)
        
        # NOTE: Main thread will handle actual hangup in cleanup
        # Transcriptions collected in _finalize_call() (finally block)

    def _perform_immediate_hangup_from_detection(self, disposition: str, intent_type: str):
        """
        Called from background detection thread when DNC/NI/HP detected.
        Creates on-demand ESL inbound connection to interrupt playback immediately.
        Falls back to existing flag-based mechanism if ESL fails.

        Thread Safety: Runs in detection thread, uses separate ESL connection.

        Args:
            disposition: "DNC", "NI", or "HP"
            intent_type: "do_not_call", "not_interested", or "hold_press"
        """
        # Step 1: ALWAYS set flags first (ensures fallback works)
        self.call_data['disposition'] = disposition
        self.call_data['intent_detected'] = intent_type
        self.is_active = False

        self._log_and_collect('warning',
            f"🚫 IMMEDIATE HANGUP TRIGGERED: {disposition} ({intent_type})")

        # Step 2: Attempt emergency ESL hangup (on-demand)
        emergency_esl = None
        try:
            # Get ESL configuration dynamically
            esl_config = ESLConfigReader.get_esl_config()

            self._log_and_collect('debug',
                f"Creating emergency ESL connection to {esl_config['host']}:{esl_config['port']}")

            # Create ESL inbound connection (on-demand)
            emergency_esl = ESL.ESLconnection(
                esl_config['host'],
                esl_config['port'],
                esl_config['password']
            )

            if not emergency_esl.connected():
                raise Exception("ESL connection failed (not connected)")

            self._log_and_collect('info', "✓ Emergency ESL connected")

            # Execute uuid_break to stop ALL audio immediately
            break_result = emergency_esl.api("uuid_break", f"{self.uuid} all")
            if break_result:
                break_response = break_result.getBody()
                self._log_and_collect('debug', f"uuid_break response: {break_response}")

            # Small delay for break to propagate
            time.sleep(0.01)  # 10ms

            # Execute uuid_kill to terminate channel
            kill_result = emergency_esl.api("uuid_kill", f"{self.uuid}")
            if kill_result:
                kill_response = kill_result.getBody()
                self._log_and_collect('debug', f"uuid_kill response: {kill_response}")

            self._log_and_collect('warning',
                f"✓ Emergency hangup executed: {disposition} (uuid_break + uuid_kill)")

        except Exception as e:
            self._log_and_collect('error',
                f"Emergency ESL hangup failed: {e} - falling back to flag-based hangup")
            # Fallback: Flags already set above, main thread will detect and handle

        finally:
            # Always clean up ESL connection
            if emergency_esl:
                try:
                    emergency_esl.disconnect()
                    self._log_and_collect('debug', "Emergency ESL disconnected")
                except Exception as e:
                    self._log_and_collect('debug', f"ESL disconnect error: {e}")

    def _execute_call_flow(self):
        """
        Execute conversation flow (SIMPLIFIED FROM OLD CODE)
        Main conversation loop
        """
        self._log_and_collect('info', f"Executing flow: {self.call_flow.get('name', 'Unknown')}")

        # Get initial step
        self.current_step = self._get_initial_step()

        # Main conversation loop
        while self.current_step != 'exit' and self.is_active:
            # Get current step
            step = self.call_flow['steps'].get(self.current_step)
            if not step:
                self._log_and_collect('error', f"Step '{self.current_step}' not found")
                break

            self._log_and_collect('info', f"Step: {self.current_step}")

            # Check if channel still exists before processing step
            if not self._is_channel_active():
                self._handle_premature_hangup()
                break

            # Process step
            try:
                if not self._process_step(step):
                    break
            except Exception as e:
                self._log_and_collect('error', f"Step processing error: {e}")
                # Preserve intent dispositions (DNC/NI/HP/etc)
                if self.call_data['disposition'] == 'INITIATED':
                    self.call_data['disposition'] = 'NP'
                break

            # Small delay between steps
            time.sleep(0.2)

        # Handle flow completion
        if self.is_active:
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = 'NP'

    def _get_initial_step(self) -> str:
        """Determine the initial step"""
        if 'steps' not in self.call_flow:
            return 'start'

        # Look for common entry points
        for potential_start in ['hello', 'introduction', 'start']:
            if potential_start in self.call_flow['steps']:
                return potential_start

        # Use first available step
        if self.call_flow['steps']:
            return list(self.call_flow['steps'].keys())[0]

        return 'start'

    def _process_step(self, step: Dict[str, Any]) -> bool:
        """
        Process a single call flow step (SIMPLIFIED FROM OLD CODE)

        Args:
            step: Step dictionary from call flow

        Returns:
            True to continue, False to end flow
        """
        # Set disposition if specified - only if not already set
        if 'disposition' in step:
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = step['disposition']
                self._log_and_collect('info', f"Disposition: {step['disposition']} (from call flow)")
            else:
                self._log_and_collect('info', f"Step specifies {step['disposition']} but preserving: {self.call_data['disposition']}")

        # Play audio if specified (using chunked playback for pause/resume on speech)
        if 'audio_file' in step:
            # Store step context for pause intent routing
            self._current_step_for_pause = step
            playback_result = self._play_audio_chunked(step['audio_file'], step)
            if playback_result == "ROUTED":
                # Step changed during playback - exit to let main loop process new step
                self._log_and_collect('debug', f"[PROCESS_STEP] Playback routed to new step: {self.current_step}")
                return True
            elif not playback_result:
                if self.call_data['disposition'] == 'INITIATED':
                    self.call_data['disposition'] = 'NP'
                return False

        # Handle action
        if 'action' in step:
            action = step['action']
            self._log_and_collect('info', f"Action: {action}")

            if action == 'transfer':
                self._handle_transfer()
                return False
            elif action == 'hangup':
                self._hangup_call()
                return False

        # Handle pause - use FreeSWITCH sleep to generate recordable silence
        if 'pause_duration' in step:
            pause_seconds = step['pause_duration']
            if pause_seconds > 0:
                # Check if channel still exists before pause
                if not self._is_channel_active():
                    self._handle_premature_hangup()
                    return False

                pause_ms = int(pause_seconds * 1000)
                self._log_and_collect('info', f"Pausing for {pause_seconds}s (using FreeSWITCH sleep for recording)")
                # FreeSWITCH sleep generates comfort noise/silence on RTP stream
                # This ensures uuid_record captures the pause as recordable silence
                self.conn.execute("sleep", str(pause_ms))

        # Handle response waiting
        if step.get('wait_for_response', False):
            return self._handle_user_response(step)
        else:
            self.current_step = step.get('next', 'exit')
            return True

    def _extract_question_from_step(self, step: Dict[str, Any]) -> str:
        """
        Extract a meaningful question from the step for Qwen context.

        Args:
            step: Current step dictionary

        Returns:
            A question string to provide context for Qwen
        """
        # Try to get question from step's audio_file name
        audio_file = step.get('audio_file', '')

        if audio_file:
            # Convert audio file name to a readable question
            # Example: "medicare_main_qualification.wav" → "Do you have Medicare?"
            # Example: "medicare_age_check.wav" → "Are you over 65?"

            question_base = audio_file.replace('.wav', '').replace('_', ' ')

            # Common question patterns based on Medicare flow
            if 'qualification' in audio_file.lower() or 'medicare' in audio_file.lower():
                return "Do you have Medicare Part A and Part B?"
            elif 'age' in audio_file.lower():
                return "Are you over the age of 65?"
            elif 'interest' in audio_file.lower():
                return "Are you interested in learning more about this?"
            else:
                # Generic question based on file name
                return f"Regarding {question_base}, how do you respond?"

        # Fallback: use step name or generic question
        step_name = self.current_step if self.current_step else 'this question'
        return f"Are you interested in continuing with {step_name}?"

    def _handle_user_response(self, step: Dict[str, Any]) -> bool:
        """
        Handle user response (SIMPLIFIED FROM OLD CODE)

        Args:
            step: Current step dictionary

        Returns:
            True to continue, False to end flow
        """
        # Stop continuous transcription before response recording to prevent bleed
        # The grace period would otherwise process user response audio as delayed playback chunks
        self.playback_periods.clear()  # Stop grace period from matching
        if self.continuous_transcription:
            self.continuous_transcription.clear_buffer()  # Clear stale audio

        timeout = step.get('timeout', 5)
        user_response = self._listen_for_response(timeout, step)

        # Handle special return values
        if user_response == "TIMEOUT_NO_RESPONSE" or not user_response:
            # True silence - no speech detected by Silero VAD
            self.conversation_log.append("User: <silence>")
            self.consecutive_silence_count += 1

            if self.consecutive_silence_count >= self.max_consecutive_silences:
                self._log_and_collect('warning', "Max silences exceeded")
                # Set DAIR 2 if not already set
                # Preserve other dispositions (NP/DNC/NI/HP/A/RI)
                # Note: DAIR is caught earlier at critical step, so reaching here means DAIR 2
                if self.call_data['disposition'] == 'INITIATED':
                    self.call_data['disposition'] = 'DAIR 2'
                else:
                    self._log_and_collect('info', f"Max silences but preserving disposition: {self.call_data['disposition']}")
                return False

            # Go to timeout retry step (using standard timeout_next field)
            self.current_step = step.get('timeout_next', step.get('next', 'exit'))
            return True

        # Reset silence counter
        self.consecutive_silence_count = 0

        # Log current call state for debugging
        self._log_and_collect('info', f"🔍 Starting intent detection for: '{user_response}'")
        self._log_and_collect('info', f"📊 Current call state: step={self.current_step}, silence_count={self.consecutive_silence_count}")

        # Pre-screen with keyword detector for explicit DNC/NI/HP (before expensive Qwen call)
        self._log_and_collect('info', "🔍 Pre-screening with keyword detector for explicit intents")
        keyword_result = self.intent_detector.detect_intent(user_response)
        if keyword_result:
            intent, confidence = keyword_result
            self._log_and_collect('info', f"📝 Keyword detector found: {intent} (confidence: {confidence:.2f})")

            # Handle explicit negative intents immediately (skip Qwen for performance)
            if intent == "do_not_call":
                self._log_and_collect('info', f"✅ Early DNC detection via keywords - skipping Qwen")
                self._handle_dnc()
                return False
            elif intent == "not_interested":
                self._log_and_collect('info', f"✅ Early NI detection via keywords - skipping Qwen")
                self._handle_not_interested()
                return False
            elif intent == "hold_press":
                self._log_and_collect('info', f"✅ Early HP detection via keywords - skipping Qwen")
                self._handle_honeypot()
                return False
            else:
                self._log_and_collect('info', f"📝 Keyword found '{intent}' but not DNC/NI/HP - proceeding to Qwen for nuanced detection")

        # Detect intent using Qwen (with fallback to keyword detector)
        intent_result = None

        # Try Qwen first if available
        if hasattr(self, 'qwen_detector') and self.qwen_detector:
            try:
                # Construct question context from step
                # Try to get a meaningful question from audio_file name or step name
                question = self._extract_question_from_step(step)

                self._log_and_collect('info', f"🤖 Attempting Qwen classification - Q: '{question}' A: '{user_response}'")

                # Get Qwen classification (returns: positive, negative, clarifying, neutral, or None)
                # Use QWEN_TOTAL_TIMEOUT to allow enough time for GPU lock wait + inference under load
                qwen_classification = self.qwen_detector.detect_intent(question, user_response, timeout=config.QWEN_TOTAL_TIMEOUT)

                if qwen_classification:
                    self._log_and_collect('info', f"✅ Qwen result: {qwen_classification}")

                    # Map Qwen classifications to next step using conditions array
                    if qwen_classification == "negative":
                        # User is declining - evaluate conditions to find next step
                        self._log_and_collect('info', "Qwen: Negative response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "negative")
                        if next_step:
                            self._log_and_collect('info', f"Qwen: Routing negative response to step: {next_step}")
                            self.current_step = next_step
                            # Reset clarification counter on successful routing
                            self.clarification_count = 0
                            # Check if the next step is a terminal "not_interested" step
                            if next_step == "not_interested":
                                self._handle_not_interested()
                                return False
                            return True
                        else:
                            # No condition matched - treat as not interested
                            self._handle_not_interested()
                            return False
                    elif qwen_classification == "positive":
                        # User is agreeing/interested - evaluate conditions to find next step
                        self._log_and_collect('info', "Qwen: Positive response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "positive")
                        if next_step:
                            self._log_and_collect('info', f"Qwen: Routing positive response to step: {next_step}")
                            self.current_step = next_step
                            # Reset clarification counter on successful routing
                            self.clarification_count = 0
                            return True
                        else:
                            # No condition matched - default to exit
                            self._log_and_collect('warning', "Qwen: No condition matched for positive response - exiting")
                            self.current_step = 'exit'
                            return False
                    elif qwen_classification == "clarifying":
                        # User needs clarification - evaluate conditions to find next step
                        self._log_and_collect('info', "Qwen: Clarifying response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "clarifying")
                        if next_step:
                            self._log_and_collect('info', f"Qwen: Routing clarifying response to step: {next_step}")
                            self.current_step = next_step
                            # Reset clarification counter on successful routing
                            self.clarification_count = 0
                            return True
                        else:
                            # No explicit clarifying route - increment clarification counter and use no_match_next
                            self._log_and_collect('info', "Qwen: No clarifying route - using no_match_next")
                            self.clarification_count += 1
                            if self.clarification_count >= self.max_clarifications:
                                self._log_and_collect('warning', "Max clarifications exceeded")
                                if self.call_data['disposition'] == 'INITIATED':
                                    self.call_data['disposition'] = 'NI'
                                else:
                                    self._log_and_collect('info', f"Max clarifications but preserving disposition: {self.call_data['disposition']}")
                                return False
                            self.current_step = step.get('no_match_next', self.current_step)
                            return True
                    elif qwen_classification == "neutral":
                        # Random/nonsensical response - evaluate conditions
                        self._log_and_collect('info', "Qwen: Neutral response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "neutral")
                        if next_step:
                            self._log_and_collect('info', f"Qwen: Routing neutral response to step: {next_step}")
                            self.current_step = next_step
                            # Reset clarification counter on successful routing
                            self.clarification_count = 0
                            return True
                        else:
                            # Treat as unclear - increment clarification counter
                            self._log_and_collect('info', "Qwen: Neutral response - treating as unclear, using no_match_next")
                            self.clarification_count += 1
                            if self.clarification_count >= self.max_clarifications:
                                self._log_and_collect('warning', "Max clarifications exceeded")
                                if self.call_data['disposition'] == 'INITIATED':
                                    self.call_data['disposition'] = 'NI'
                                else:
                                    self._log_and_collect('info', f"Max clarifications but preserving disposition: {self.call_data['disposition']}")
                                return False
                            self.current_step = step.get('no_match_next', self.current_step)
                            return True
                    else:
                        # Handle other intents (rebuttal_question_*, unhandled_question, etc.)
                        self._log_and_collect('info', f"Qwen: {qwen_classification} detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, qwen_classification)

                        # Increment clarification counter for rebuttals (they count as needing clarification)
                        self.clarification_count += 1
                        self._log_and_collect('info', f"Qwen: Clarification count: {self.clarification_count}/{self.max_clarifications}")

                        if self.clarification_count >= self.max_clarifications:
                            self._log_and_collect('warning', "Max clarifications exceeded")
                            if self.call_data['disposition'] == 'INITIATED':
                                self.call_data['disposition'] = 'NI'
                            else:
                                self._log_and_collect('info', f"Max clarifications but preserving disposition: {self.call_data['disposition']}")
                            return False

                        if next_step:
                            self._log_and_collect('info', f"Qwen: Routing {qwen_classification} to step: {next_step}")
                            self.current_step = next_step
                            return True
                        else:
                            # No condition matched - use no_match_next
                            self._log_and_collect('info', f"Qwen: No route for {qwen_classification} - using no_match_next")
                            self.current_step = step.get('no_match_next', self.current_step)
                            return True
                else:
                    # Qwen returned None (timeout or error) - fall back to keyword detector
                    self._log_and_collect('warning', f"⚠️ Qwen returned None (likely timeout) - falling back to keyword detector")
                    # Log Qwen metrics for debugging
                    try:
                        metrics = self.qwen_detector.get_performance_metrics()
                        self._log_and_collect('info', f"📈 Qwen metrics: {metrics}")
                    except:
                        pass

            except Exception as e:
                self._log_and_collect('warning', f"⚠️ Qwen detection exception: {e} - falling back to keyword detector")

        # Fallback to keyword-based intent detector
        self._log_and_collect('info', "📝 Using keyword-based intent detector as fallback")
        result = self.intent_detector.detect_intent(user_response)
        if result:
            intent, confidence = result
            self._log_and_collect('info', f"✅ Keyword detector SUCCEEDED: {intent} (confidence: {confidence:.2f})")

            # Handle intents
            if intent == "do_not_call":
                self._handle_dnc()
                return False
            elif intent == "not_interested":
                self._handle_not_interested()
                return False
            elif intent == "qualified":
                self._handle_qualified()
                return False

            # If keyword detector found some other intent, reset clarification counter
            self.clarification_count = 0
        else:
            self._log_and_collect('info', f"❌ Keyword detector found NO MATCH for: '{user_response}'")

        # No specific intent detected - increment clarification counter and use no_match_next
        self._log_and_collect('info', "No specific intent detected - using no_match_next for clarification")
        self.clarification_count += 1

        if self.clarification_count >= self.max_clarifications:
            self._log_and_collect('warning', "Max clarifications exceeded")
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = 'NI'
            else:
                self._log_and_collect('info', f"Max clarifications but preserving disposition: {self.call_data['disposition']}")
            return False

        self.current_step = step.get('no_match_next', self.current_step)
        return True

    def _handle_dnc(self):
        """Handle Do Not Call request"""
        self._log_and_collect('warning', "🚫 DNC request detected - ending call immediately")
        self._perform_hangup_immediate('DNC', 'do_not_call', 'dnc_from_response')

    # OBSOLETE METHODS REMOVED (replaced by _check_all_detections() unified approach):
    # - _handle_dnc_detected_during_playback()
    # - _handle_ni_detected_during_playback()
    # - _handle_hp_detected_during_playback()
    # All playback interruptions now handled by unified detection checker in _play_audio() loop

    def _map_qwen_intent_to_step(self, step, qwen_intent):
        """Map Qwen intent (positive/negative/clarifying) to next step based on conditions array"""
        conditions = step.get('conditions', [])

        if not conditions:
            return step.get('next', 'exit')

        # First priority: Check for intent-type conditions (new format)
        for condition in conditions:
            if condition.get('type') == 'intent' and condition.get('intent') == qwen_intent:
                next_step = condition.get('next')
                self._log_and_collect('debug', f"Matched intent condition: {qwen_intent} -> {next_step}")
                return next_step

        # Handle clarifying intent specifically
        if qwen_intent == "clarifying":
            # User is asking for more information
            # Look for a condition with intent='clarifying' in the step
            for condition in conditions:
                if condition.get('type') == 'intent' and condition.get('intent') == 'clarifying':
                    next_step = condition.get('next')
                    self._log_and_collect('debug', f"Qwen classified as clarifying - routing to: {next_step}")
                    return next_step

            # If no explicit clarifying condition, treat as neutral (trigger clarification)
            self._log_and_collect('debug', f"Qwen classified as clarifying but no route - triggering clarification")
            return None

        # Fallback: Handle legacy keyword-based conditions
        positive_keywords = ['yes', 'yeah', 'i do', 'i have', 'sure', 'okay', 'correct']
        negative_keywords = ['no', 'nope', "don't", "not", "none", "neither"]

        if qwen_intent == "positive":
            # Find condition with positive keywords
            for condition in conditions:
                if condition.get('type') == 'contains':
                    keywords = [kw.lower() for kw in condition.get('keywords', [])]
                    if any(kw in positive_keywords for kw in keywords):
                        return condition.get('next')

            # If no explicit positive condition, take first condition
            if conditions:
                return conditions[0].get('next')

        elif qwen_intent == "negative":
            # Find condition with negative keywords
            for condition in conditions:
                if condition.get('type') == 'contains':
                    keywords = [kw.lower() for kw in condition.get('keywords', [])]
                    if any(kw in negative_keywords for kw in keywords):
                        return condition.get('next')

            # If no explicit negative condition, take second condition if exists
            if len(conditions) > 1:
                return conditions[1].get('next')

        elif qwen_intent == "neutral":
            # Neutral responses should trigger clarification
            # Return None to use no_match_next path (clarification)
            self._log_and_collect('debug', f"Qwen classified response as neutral - triggering clarification")
            return None

        # No matching condition found - return None to use no_match_next
        return None

    def _handle_not_interested(self):
        """Handle Not Interested response"""
        self._log_and_collect('warning', "⚠️ Not interested detected - ending call immediately")
        self._perform_hangup_immediate('NI', 'not_interested', 'ni_from_response')

    def _handle_honeypot(self):
        """Handle honeypot (hold/press) detection"""
        self._log_and_collect('warning', "🍯 Honeypot detected - ending call immediately")
        self._perform_hangup_immediate('HP', 'hold_press', 'hp_from_response')

    def _handle_qualified(self):
        """Handle qualified lead"""
        self._log_and_collect('info', "✅ Qualified lead detected")
        # Only set SALE if disposition not already set
        if self.call_data['disposition'] == 'INITIATED':
            self.call_data['disposition'] = 'SALE'
            self.call_data['intent_detected'] = 'qualified'
        else:
            self._log_and_collect('info', f"Qualified detected but preserving disposition: {self.call_data['disposition']}")
        # Transfer will be handled by step action

    def _handle_transfer(self):
        """
        Handle qualified transfer action (terminal step: action='transfer').
        Replicates pjsua2's _handle_transfer_safe functionality.
        """
        did = self.agent_config.did_transfer_qualified
        if not did:
            self._log_and_collect('error', "No qualified transfer DID configured")
            # Fall back to non-transfer qualified disposition
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = 'QUALIFIED'
            return

        self._log_and_collect('info', f"🔄 Initiating qualified transfer to {did}")

        # Set disposition to SALE before transfer (if not already set)
        if self.call_data['disposition'] == 'INITIATED':
            self.call_data['disposition'] = 'SALE'

        # Attempt qualified transfer
        if self._transfer_call(did, is_qualified=True):
            self._log_and_collect('info', "✅ Qualified transfer successful")
            self.call_data['call_result'] = 'transferred_qualified'
        else:
            self._log_and_collect('error', "❌ Qualified transfer failed")
            # Set NP (No Presentation) if disposition is SALE but transfer failed
            if self.call_data['disposition'] == 'SALE':
                self.call_data['disposition'] = 'NP'
            self.call_data['call_result'] = 'transfer_failed'

    def _collect_continuous_transcriptions(self):
        """
        Collect transcriptions from continuous listener and add to conversation log
        Similar to old pjsua2 implementation
        """
        if not self.continuous_transcription:
            return

        try:
            # Get all transcriptions since call start
            since_time = self.call_data['start_time']
            transcriptions = self.continuous_transcription.get_transcriptions_since(
                since_time,
                min_confidence=0.3
            )

            if transcriptions:
                self._log_and_collect('info', f"Collected {len(transcriptions)} continuous transcriptions for logging")

                # Add to conversation log with timestamps
                for timestamp, text, confidence in transcriptions:
                    # Calculate time offset from call start
                    offset = timestamp - self.call_data['start_time']
                    time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))

                    # Format similar to old implementation
                    formatted_entry = f"User (continuous @ {time_str}): {text}"
                    self.conversation_log.append(formatted_entry)

                    self._log_and_collect('debug', f"Added continuous transcription: {formatted_entry}")

        except Exception as e:
            self._log_and_collect('error', f"Error collecting continuous transcriptions: {e}", exc_info=True)

    def _save_full_call_recording(self):
        """
        Stop full call recording and save to CRM uploads directory
        This saves the complete stereo recording (caller + bot) to /var/www/recordings/
        """
        try:
            if not hasattr(self, 'full_recording_file') or not self.full_recording_file:
                self._log_and_collect('warning', "No full recording file to save")
                return

            # Stop recording
            cmd = f"uuid_record {self.uuid} stop {self.full_recording_file}"
            try:
                result = self.conn.api(cmd)
                if result:
                    self._log_and_collect('info', f"Full call recording stopped: {result.getBody()}")
                else:
                    self._log_and_collect('warning', "No response from uuid_record stop command")
            except Exception as e:
                self._log_and_collect('warning', f"Error stopping recording: {e}")

            # Wait briefly for file to be fully written
            time.sleep(0.2)

            # Check if recording file exists and has content
            if not os.path.exists(self.full_recording_file):
                self._log_and_collect('error', f"Recording file not found: {self.full_recording_file}")
                return

            file_size = os.path.getsize(self.full_recording_file)
            if file_size < 1024:  # Less than 1KB
                self._log_and_collect('warning', f"Recording file too small ({file_size} bytes) - may be empty")
                return

            # Create final path in CRM upload directory
            call_id = self.call_data.get('id') or self.uuid
            final_path = os.path.join(SUITECRM_UPLOAD_DIR, f"{call_id}")

            # Ensure directory exists
            os.makedirs(SUITECRM_UPLOAD_DIR, exist_ok=True)

            # Move file to CRM upload directory
            import shutil
            shutil.move(self.full_recording_file, final_path)

            # Update call_data for CRM logging
            self.call_data['filename'] = f"{call_id}"
            self.call_data['file_mime_type'] = 'audio/wav'

            self._log_and_collect('info', f"✅ Full call recording saved: {final_path} ({file_size:,} bytes)")
            self._log_and_collect('info', f"  Recording contains: entire call from answer to hangup")
            self._log_and_collect('info', f"  Stereo: LEFT channel = caller, RIGHT channel = bot")
            self._log_and_collect('info', f"  Database fields updated: filename={self.call_data['filename']}, mime_type={self.call_data['file_mime_type']}")

        except Exception as e:
            self._log_and_collect('error', f"Error saving full call recording: {e}", exc_info=True)

    def _finalize_call(self):
        """Finalize call and save to CRM"""
        try:
            # Collect any remaining continuous transcriptions
            self._collect_continuous_transcriptions()

            # Calculate duration
            self.call_data['end_time'] = time.time()
            self.call_data['duration'] = int(
                self.call_data['end_time'] - self.call_data['start_time']
            )

            # Build transcript
            self.call_data['transcript'] = "\n".join(self.conversation_log)

            # Collect call logs for database storage
            if hasattr(self, 'call_log_collector'):
                self.call_data['call_logs'] = self.call_log_collector.get_logs()

            # Log continuous transcription stats
            if self.continuous_transcription:
                stats = self.continuous_transcription.get_stats()
                self._log_and_collect('info', 
                    f"Continuous transcription stats: "
                    f"chunks={stats['chunks_processed']}, "
                    f"transcriptions={stats['transcriptions_successful']}/{stats['transcriptions_attempted']}, "
                    f"dnc={stats['dnc_detections']}, ni={stats['ni_detections']}, hp={stats['hp_detections']}"
                )

            # Stop and save full call recording (populates filename and file_mime_type in call_data)
            self._save_full_call_recording()

            # Save to CRM
            if self.crm_logger:
                success = self.crm_logger.log_call_end(self.call_data)
                if success:
                    self.crm_updated = True
                    self._log_and_collect('info', "✅ Call saved to CRM")
                else:
                    self._log_and_collect('error', "❌ Failed to save call to CRM")

            # Process call outcome and report to ViciDial
            if self.outcome_handler:
                try:
                    outcome = self._determine_call_outcome()
                    self.outcome_handler.process_call_outcome(
                        self.phone_number,
                        outcome,
                        self.call_data,
                        None  # intent_data not used in current implementation
                    )
                except Exception as e:
                    self._log_and_collect('error', f"Outcome processing failed: {e}")

            self._log_and_collect('info', f"Call finalized: Disposition={self.call_data['disposition']}, "
                           f"Duration={self.call_data['duration']}s")

        except Exception as e:
            self._log_and_collect('error', f"Finalization error: {e}")

    def _cleanup_continuous_transcription(self):
        """Cleanup continuous transcription handler and free memory"""
        if self.continuous_transcription:
            try:
                # Clear audio buffer (can be up to 2 MB)
                with self.continuous_transcription.buffer_lock:
                    self.continuous_transcription.audio_buffer.clear()
                    self._log_and_collect('debug', "[CLEANUP] Audio buffer cleared")

                # Clear transcriptions list
                with self.continuous_transcription.transcription_lock:
                    self.continuous_transcription.transcriptions.clear()
                    self._log_and_collect('debug', "[CLEANUP] Transcriptions cleared")

                # Clear playback periods
                with self.continuous_transcription.playback_lock:
                    self.continuous_transcription.playback_periods.clear()

                # Release VAD singleton reference (model persists globally)
                if hasattr(self.continuous_transcription, 'silero_vad_singleton'):
                    self.continuous_transcription.silero_vad_singleton = None

                # Clear reference
                self.continuous_transcription = None

                self._log_and_collect('info', "[CLEANUP] Continuous transcription handler freed")
            except Exception as e:
                self._log_and_collect('error', f"[CLEANUP] Error cleaning continuous transcription: {e}")

    def _cleanup_detection_thread(self):
        """Cleanup detection thread with extended timeout"""
        try:
            # Signal stop
            self.detection_stop_event.set()

            # Wait for thread to finish (increased timeout from 1.0s to 3.0s)
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=3.0)

                if self.detection_thread.is_alive():
                    self._log_and_collect('warning', "[CLEANUP] Detection thread did not stop within timeout")

            # Clear reference
            self.detection_thread = None

            self._log_and_collect('info', "[CLEANUP] Detection thread cleaned up")
        except Exception as e:
            self._log_and_collect('error', f"[CLEANUP] Error cleaning detection thread: {e}")

    def _cleanup_conversation_log(self):
        """Clear conversation log to free memory"""
        try:
            if hasattr(self, 'conversation_log') and self.conversation_log:
                log_size = len(self.conversation_log)
                self.conversation_log.clear()
                self._log_and_collect('debug', f"[CLEANUP] Conversation log cleared ({log_size} entries)")
        except Exception as e:
            self._log_and_collect('error', f"[CLEANUP] Error clearing conversation log: {e}")

    def _cleanup_playback_periods(self):
        """Clear playback period tracking"""
        try:
            if hasattr(self, 'playback_periods') and self.playback_periods:
                self.playback_periods.clear()
                self._log_and_collect('debug', "[CLEANUP] Playback periods cleared")
        except Exception as e:
            self._log_and_collect('error', f"[CLEANUP] Error clearing playback periods: {e}")

    def _cleanup_temp_files(self):
        """
        Cleanup all temp recording files with retry logic
        FreeSWITCH may hold file handles open - need aggressive retry strategy
        """
        if not hasattr(self, 'temp_files') or not self.temp_files:
            return

        with self.temp_files_lock:
            files_to_clean = self.temp_files.copy()

        cleaned = 0
        failed = 0

        for temp_file in files_to_clean:
            if not os.path.exists(temp_file):
                cleaned += 1
                continue

            # Try immediate deletion first
            try:
                os.unlink(temp_file)
                cleaned += 1
                self._log_and_collect('debug', f"[CLEANUP] Deleted temp file: {temp_file}")
                continue
            except Exception as e:
                self._log_and_collect('debug', f"[CLEANUP] Immediate delete failed for {temp_file}: {e}")

            # Retry with delays (FreeSWITCH may still have handle open)
            deleted = False
            for attempt in range(3):
                try:
                    time.sleep(0.1 * (attempt + 1))  # 0.1s, 0.2s, 0.3s delays
                    os.unlink(temp_file)
                    cleaned += 1
                    deleted = True
                    self._log_and_collect('debug', f"[CLEANUP] Deleted temp file on retry {attempt+1}: {temp_file}")
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        self._log_and_collect('warning', f"[CLEANUP] Failed to delete {temp_file} after 3 attempts: {e}")
                        failed += 1

        # Clear the list
        with self.temp_files_lock:
            self.temp_files.clear()

        if cleaned > 0 or failed > 0:
            self._log_and_collect('info', f"[CLEANUP] Temp files: {cleaned} deleted, {failed} failed")

    def _cleanup(self):
        """Cleanup all resources using resource manager"""
        try:
            self.is_active = False

            # Fallback CRM logging if _finalize_call was not called or failed
            if self.crm_logger and not self.crm_updated:
                try:
                    self._log_and_collect('info', "[CLEANUP] Performing fallback CRM logging")
                    # Ensure call data has required fields
                    if not self.call_data.get('end_time'):
                        self.call_data['end_time'] = time.time()
                    if not self.call_data.get('duration'):
                        self.call_data['duration'] = int(
                            self.call_data['end_time'] - self.call_data['start_time']
                        )
                    if not self.call_data.get('transcript'):
                        self.call_data['transcript'] = "\n".join(self.conversation_log)
                    # Collect call logs if not already done
                    if hasattr(self, 'call_log_collector') and not self.call_data.get('call_logs'):
                        self.call_data['call_logs'] = self.call_log_collector.get_logs()

                    success = self.crm_logger.log_call_end(self.call_data)
                    if success:
                        self.crm_updated = True
                        self._log_and_collect('info', "[CLEANUP] ✅ Fallback CRM logging successful")

                        # CRITICAL FIX: Also report to ViciDial in fallback path
                        if self.outcome_handler:
                            try:
                                self._log_and_collect('info', "[CLEANUP] Performing fallback ViciDial reporting")
                                outcome = self._determine_call_outcome()
                                self.outcome_handler.process_call_outcome(
                                    self.phone_number,
                                    outcome,
                                    self.call_data,
                                    None
                                )
                                self._log_and_collect('info', "[CLEANUP] ✅ Fallback ViciDial reporting successful")
                            except Exception as vicidial_e:
                                self._log_and_collect('error', f"[CLEANUP] ViciDial reporting error: {vicidial_e}")
                    else:
                        self._log_and_collect('error', "[CLEANUP] ❌ Fallback CRM logging failed")
                except Exception as e:
                    self._log_and_collect('error', f"[CLEANUP] Fallback CRM logging error: {e}")

            # Stop audio detection first (critical path)
            self._stop_audio_detection()

            # Cleanup all registered resources via resource manager
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup_all()

            # Force garbage collection to free memory immediately
            import gc
            gc.collect()

            self._log_and_collect('info', "[CLEANUP] Complete - all resources released")
        except Exception as e:
            self._log_and_collect('error', f"[CLEANUP] Error: {e}", exc_info=True)
