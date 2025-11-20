#!/usr/bin/env python3
"""
FreeSWITCH ESL Bot Server
- Simple TCP socket server listening on 127.0.0.1:8084
- Accepts incoming connections from FreeSWITCH (outbound socket mode)
- Spawns handler thread per call
- No complex threading or PJSIP management
"""

import socket
import threading
import logging
import sys
import signal
import time
import random
import weakref
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# Import ESL
try:
    import ESL
except ImportError:
    print("ERROR: ESL module not found. Install with:")
    print("  cd /usr/src/freeswitch/libs/esl")
    print("  make pymod && make pymod-install")
    sys.exit(1)

# Import bot components
from freeswitch_bot_handler import FreeSWITCHBotHandler
from src.suitecrm_integration import fetch_active_agent_configs, SuiteCRMAgentConfig, CentralizedDBManager
from src.server_id_service import server_id_service
from src.hp_disposition_cache import get_hp_cache

# Import model singletons for preloading (standalone modules)
from src.parakeet_singleton import ParakeetModelSingleton
from src.silero_vad_singleton import SileroVADSingleton
try:
    from src.qwen_singleton import QwenModelSingleton
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    QwenModelSingleton = None

# Configuration
LOG_DIR = "/var/log/sip-bot"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8084
MAX_CONNECTIONS = 1000  # Maximum concurrent calls

# Global state
SHUTTING_DOWN = threading.Event()
active_calls = weakref.WeakValueDictionary()  # Track active calls {uuid: handler} - auto-cleanup on GC
active_calls_lock = threading.Lock()

# Model singletons (initialized at startup)
parakeet_singleton = None
qwen_singleton = None
silero_singleton = None

# Statistics
stats = {
    'total_calls': 0,
    'active_calls': 0,
    'accepted_calls': 0,
    'rejected_calls': 0,
    'failed_calls': 0
}
stats_lock = threading.Lock()


def periodic_cleanup():
    """Monitor and force-cleanup stuck calls"""
    while not SHUTTING_DOWN.is_set():
        time.sleep(30)  # Check every 30 seconds

        with active_calls_lock:
            current_time = time.time()
            stuck_calls = []

            for uuid, handler in list(active_calls.items()):
                try:
                    age = current_time - handler.call_start_time

                    # Only kill if truly stuck (> 5 minutes + no progress)
                    if age > 300:  # 5 minutes
                        # Check if making progress
                        if hasattr(handler, 'current_step') and handler.current_step and handler.is_active:
                            # Call is progressing through script, leave it alone
                            continue
                        # Only kill if truly stuck (no progress or inactive)
                        stuck_calls.append((uuid, handler, age))
                except Exception as e:
                    log.debug(f"Error checking call {uuid}: {e}")

            # Force cleanup stuck calls
            for uuid, handler, age in stuck_calls:
                log.warning(f"Force cleaning stuck call {uuid} (age={age:.0f}s)")
                handler.is_active = False  # Signal to stop (also stops detection thread)
                active_calls.pop(uuid, None)

                # Update stats
                with stats_lock:
                    stats['active_calls'] = len(active_calls)

                # Force cleanup resources
                try:
                    handler.resource_manager.cleanup_all()
                except Exception as e:
                    log.debug(f"Error cleaning resources for {uuid}: {e}")

                # Close ESL connection
                try:
                    handler.conn.disconnect()
                except Exception as e:
                    log.debug(f"Error disconnecting ESL for {uuid}: {e}")


# --- Logging Setup ---
# Only configure logging once (avoid duplicate handlers when module is imported)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO to avoid excessive debug output from libraries
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(f'{LOG_DIR}/bot_server.log'),
            logging.StreamHandler()  # Also output to stdout for journalctl
        ]
    )
log = logging.getLogger('BotServer')
log.propagate = False  # Prevent double logging through root propagation

# Suppress noisy third-party library debug logs
# These ML libraries produce excessive debug output that clutters the logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('nemo').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    log.warning(f"Shutdown signal {signum} received")
    SHUTTING_DOWN.set()
    sys.exit(0)


def select_agent(esl_info: ESL.ESLevent, agent_configs: List[SuiteCRMAgentConfig]) -> Optional[SuiteCRMAgentConfig]:
    """
    Select appropriate agent based on call information

    Args:
        esl_info: ESL event with call information
        agent_configs: List of available agent configurations

    Returns:
        Selected agent config or None
    """
    # Extract source IP from FreeSWITCH
    source_ip = esl_info.getHeader("variable_sip_h_X-FS-Support")  # Custom header
    if not source_ip:
        # Fallback to network address
        source_ip = esl_info.getHeader("variable_sip_network_ip")

    # Extract campaign ID from custom header (ViciDial passes this)
    campaign_id = esl_info.getHeader("variable_sip_h_X-VICIdial-Campaign-Id")

    log.info(f"Selecting agent for source_ip={source_ip}, campaign_id={campaign_id}")

    # Build set of valid server IPs
    valid_server_ips = set()
    for cfg in agent_configs:
        if cfg.server_ip:
            ips = [ip.strip() for ip in cfg.server_ip.split(',') if ip.strip()]
            valid_server_ips.update(ips)

    # Filter agents by server IP (if we have source_ip)
    if source_ip:
        matching_agents = [
            cfg for cfg in agent_configs
            if cfg.server_ip and source_ip in [ip.strip() for ip in cfg.server_ip.split(',')]
        ]
    else:
        # If no source IP, use all agents
        matching_agents = agent_configs

    if not matching_agents:
        log.warning(f"No agents found for source_ip={source_ip}")
        return None

    # Random selection for load balancing
    selected = random.choice(matching_agents)
    log.info(f"Selected agent: {selected.agent_id[:8]}")

    return selected


def log_hp_rejection(phone_number: str, vici_lead_id: str = None, vici_list_id: str = None) -> Optional[str]:
    """
    Log HP rejection to database (minimal record)

    Args:
        phone_number: The rejected phone number
        vici_lead_id: ViciDial lead ID from SIP headers
        vici_list_id: ViciDial list ID from SIP headers

    Returns:
        call_id on success, None on failure
    """
    try:
        call_id = str(uuid.uuid4())
        now_utc = datetime.utcnow()
        server_id = server_id_service.get_server_id()

        query = """
        INSERT INTO e_call (
            id, name, date_entered, date_modified, disposition,
            vici_lead_id, vici_list_id, start_date_time, end_date_time,
            sh_ep_servers_id, description, deleted
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        timestamp = now_utc.strftime('%Y-%m-%d %H:%M:%S')
        params = (
            call_id,
            phone_number,
            timestamp,
            timestamp,
            'HP',  # Disposition indicates HP rejection
            vici_lead_id,
            vici_list_id,
            timestamp,
            timestamp,  # Instant rejection
            server_id,
            'HP Cache Rejection - Prior HP disposition found',
            0
        )

        result = CentralizedDBManager.execute_query(
            query, params, context="hp_rejection", logger=log
        )

        if result is not None:
            log.info(f"HP rejection logged to database: call_id={call_id}")
            return call_id
        else:
            log.error(f"Failed to log HP rejection for {phone_number}")
            return None

    except Exception as e:
        log.error(f"Error logging HP rejection: {e}", exc_info=True)
        return None


def handle_call(conn_socket: socket.socket, agent_configs: List[SuiteCRMAgentConfig]):
    """
    Handle incoming call from FreeSWITCH

    Args:
        conn_socket: Socket connection from FreeSWITCH
        agent_configs: List of available agent configurations
    """
    esl_conn = None
    call_uuid = None

    try:
        # Create ESL connection from socket file descriptor
        fd = conn_socket.fileno()
        esl_conn = ESL.ESLconnection(fd)

        if not esl_conn.connected():
            log.error("ESL connection failed")
            with stats_lock:
                stats['failed_calls'] += 1
            return

        # Get call information
        info = esl_conn.getInfo()
        call_uuid = info.getHeader("Unique-ID")
        caller_id = info.getHeader("Caller-Caller-ID-Number")
        destination = info.getHeader("Caller-Destination-Number")

        log.info(f"📞 New call: UUID={call_uuid}, From={caller_id}, To={destination}")

        # === HP CACHE CHECK - EARLY REJECTION ===
        hp_cache = get_hp_cache()
        if hp_cache and hp_cache.is_ready():
            if hp_cache.check_hp_disposition(caller_id):
                log.warning(f"🚫 HP REJECTION: {caller_id} found in HP cache - rejecting call")

                # Extract ViciDial metadata for logging
                vici_lead_id = info.getHeader("variable_sip_h_X-VICIdial-Lead-Id")
                vici_list_id = info.getHeader("variable_sip_h_X-VICIdial-List-Id")

                # Log rejection to database
                log_hp_rejection(caller_id, vici_lead_id, vici_list_id)

                # Reject call with CALL_REJECTED (SIP 603)
                esl_conn.execute("hangup", "CALL_REJECTED")

                with stats_lock:
                    stats['total_calls'] += 1
                    stats['rejected_calls'] += 1

                return  # Exit without processing call
        # === END HP CACHE CHECK ===

        with stats_lock:
            stats['total_calls'] += 1
            stats['active_calls'] += 1
            stats['accepted_calls'] += 1

        # Select agent
        agent_config = select_agent(info, agent_configs)
        if not agent_config:
            log.error(f"No agent available for call {call_uuid}")
            esl_conn.execute("hangup", "")
            with stats_lock:
                stats['rejected_calls'] += 1
            return

        # Extract ViciDial metadata from SIP headers
        vici_lead_id = info.getHeader("variable_sip_h_X-VICIdial-Lead-Id")
        vici_list_id = info.getHeader("variable_sip_h_X-VICIdial-List-Id")
        vici_campaign_id = info.getHeader("variable_sip_h_X-VICIdial-Campaign-Id")

        log.info(f"ViciDial metadata: lead_id={vici_lead_id}, list_id={vici_list_id}, campaign_id={vici_campaign_id}")

        # Create handler
        handler = FreeSWITCHBotHandler(
            esl_conn,
            agent_config,
            info,
            vici_lead_id=vici_lead_id,
            vici_list_id=vici_list_id,
            vici_campaign_id=vici_campaign_id
        )

        # Track active call
        with active_calls_lock:
            active_calls[call_uuid] = handler

        # Run call handler (blocks until call completes)
        handler.run()

        log.info(f"✅ Call {call_uuid} completed")

    except Exception as e:
        log.error(f"Error handling call {call_uuid}: {e}", exc_info=True)
        with stats_lock:
            stats['failed_calls'] += 1

    finally:
        # Cleanup
        if call_uuid:
            with active_calls_lock:
                active_calls.pop(call_uuid, None)

        with stats_lock:
            stats['active_calls'] -= 1

        # Disconnect ESL connection explicitly (prevents connection leaks)
        if esl_conn:
            try:
                esl_conn.disconnect()
            except:
                pass

        if conn_socket:
            try:
                conn_socket.close()
            except:
                pass


def log_stats():
    """Periodically log statistics"""
    while not SHUTTING_DOWN.is_set():
        time.sleep(30)  # Every 30 seconds

        with stats_lock:
            log.info(f"📊 Stats: Total={stats['total_calls']}, "
                    f"Active={stats['active_calls']}, "
                    f"Accepted={stats['accepted_calls']}, "
                    f"Rejected={stats['rejected_calls']}, "
                    f"Failed={stats['failed_calls']}")


def main():
    """Main server entry point"""
    # Configure system resource limits (file descriptors)
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        log.info(f"Current FD limits: soft={soft}, hard={hard}")

        # Try to set to 65535, fall back to hard limit if not possible
        target_limit = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, target_limit))

        # Verify what we actually got
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        log.info(f"✅ FD limits set to: soft={new_soft}, hard={new_hard}")
    except Exception as e:
        log.warning(f"⚠️ Could not increase FD limit: {e}")
        log.warning("Consider setting LimitNOFILE=65535 in systemd service")

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    log.info("="*60)
    log.info("🚀 FreeSWITCH ESL Bot Server Starting")
    log.info("="*60)

    try:
        # Retrieve server ID
        log.info("🔗 Retrieving server ID from centralized API...")
        server_id = server_id_service.retrieve_server_id(max_retries=3)
        if server_id:
            log.info(f"✅ Server ID: {server_id}")
        else:
            log.warning("⚠️ Server ID retrieval failed - calls will log without server ID")

        # Load agent configurations
        log.info("📋 Loading agent configurations...")
        agent_configs = fetch_active_agent_configs()

        if not agent_configs:
            log.critical("❌ No active agents found in database")
            return 1

        log.info(f"✅ Loaded {len(agent_configs)} agents")

        # Extract global RNNT confidence threshold from first agent config
        # All agents on a server share the same campaign threshold (verified via DB query)
        global_rnnt_confidence_threshold = 0  # Default fallback

        if agent_configs:
            first_config = agent_configs[0]
            global_rnnt_confidence_threshold = first_config.rnnt_confidence_threshold
            log.info(f"📊 Global RNNT confidence threshold from database: {global_rnnt_confidence_threshold}")
        else:
            log.info(f"📊 Using default RNNT confidence threshold: {global_rnnt_confidence_threshold}")

        # Preload ML models at startup (before accepting calls)
        log.info("🤖 Preloading ML models...")
        global parakeet_singleton, qwen_singleton, silero_singleton

        try:
            log.info("  Loading Parakeet RNNT model...")
            parakeet_singleton = ParakeetModelSingleton()
            # Trigger model load with database threshold
            _ = parakeet_singleton.get_model(log, confidence_threshold=global_rnnt_confidence_threshold)
            log.info("  ✅ Parakeet model loaded and ready")
        except Exception as e:
            log.error(f"  ❌ Failed to preload Parakeet: {e}")
            return 1

        if QWEN_AVAILABLE:
            try:
                log.info("  Loading Qwen intent detector...")
                qwen_singleton = QwenModelSingleton.get_instance()
                # Trigger model load by getting it once
                _ = qwen_singleton.get_detector(log)
                log.info("  ✅ Qwen model loaded and ready")
            except Exception as e:
                log.warning(f"  ⚠️ Failed to preload Qwen: {e}")
                qwen_singleton = None

        # Preload Silero VAD model (CPU, fast)
        try:
            log.info("  Loading Silero VAD model (CPU)...")
            silero_singleton = SileroVADSingleton()
            model = silero_singleton.get_model(log)
            if model:
                log.info("  ✅ Silero VAD loaded on CPU (2.2MB, 0.11ms inference)")
            else:
                log.warning("  ⚠️ Silero VAD model not available")
        except Exception as e:
            log.warning(f"  ⚠️ Failed to preload Silero VAD: {e}")
            log.warning("  Calls will load on-demand (slower first call)")

        log.info("✅ All models preloaded")

        # Pre-load HP Disposition Cache (all historical records, one-time only)
        log.info("🔄 Loading HP Disposition Cache...")
        try:
            hp_cache = get_hp_cache()
            hp_cache.start_preload(blocking=True, shutdown_event=SHUTTING_DOWN, refresh=False)
            cache_stats = hp_cache.get_stats()
            log.info(f"✅ HP Cache ready: {cache_stats['cache_size']} numbers loaded")
        except Exception as e:
            log.warning(f"⚠️ Failed to preload HP Cache: {e}")
            log.warning("  HP cache checks will be skipped")

        log.info("✅ Ready to accept calls")

        # Start statistics logging thread
        stats_thread = threading.Thread(target=log_stats, daemon=True)
        stats_thread.start()

        # Start cleanup monitor thread
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
        log.info("✅ Cleanup monitor started (checks every 30s, cleans calls >5min)")

        # Create TCP server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((LISTEN_HOST, LISTEN_PORT))
        server_socket.listen(MAX_CONNECTIONS)

        log.info(f"🎧 Listening on {LISTEN_HOST}:{LISTEN_PORT}")
        log.info("="*60)
        log.info("✅ Bot Server Ready - Waiting for FreeSWITCH connections...")
        log.info("="*60)

        # Main accept loop
        while not SHUTTING_DOWN.is_set():
            try:
                # Accept connection with timeout
                server_socket.settimeout(1.0)

                try:
                    conn_socket, addr = server_socket.accept()
                except socket.timeout:
                    continue

                log.info(f"📥 Connection from {addr}")

                # Spawn handler thread
                handler_thread = threading.Thread(
                    target=handle_call,
                    args=(conn_socket, agent_configs),
                    daemon=True
                )
                handler_thread.start()

            except KeyboardInterrupt:
                log.info("Keyboard interrupt received")
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)

        log.info("Shutting down...")

    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup HP cache
        try:
            hp_cache = get_hp_cache()
            hp_cache.shutdown(timeout=5)
            log.info("✅ HP Cache shutdown complete")
        except Exception as e:
            log.debug(f"HP cache cleanup: {e}")

        # Cleanup server socket
        if 'server_socket' in locals():
            server_socket.close()

        log.info("✅ Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
