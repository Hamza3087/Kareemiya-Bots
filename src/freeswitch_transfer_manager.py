"""
FreeSWITCH Transfer Manager

Handles SIP transfers using FreeSWITCH's uuid_deflect command (SIP REFER).
Replicates the pjsua2 transfer functionality for the FreeSWITCH implementation.

Based on: src/sip_transfer_manager.py (pjsua2 version)
"""

import time
import logging
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TransferStatus(Enum):
    """Transfer status enumeration"""
    NOT_ATTEMPTED = "not_attempted"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ALREADY_TRANSFERRED = "already_transferred"


class TransferRecord:
    """Record of a transfer attempt"""
    def __init__(self, uuid: str, did: str, target_ip: str):
        self.uuid = uuid
        self.did = did
        self.target_ip = target_ip
        self.timestamp = datetime.now()
        self.attempt_count = 0
        self.status = TransferStatus.NOT_ATTEMPTED
        self.error_message: Optional[str] = None


class ThreadSafeTransferTracker:
    """
    Thread-safe tracker to prevent duplicate transfers.
    Similar to pjsua2 implementation's ThreadSafeTransferManager.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._transferred_uuids: Dict[str, TransferRecord] = {}

    def is_already_transferred(self, uuid: str) -> bool:
        """Check if a call UUID has already been transferred"""
        with self._lock:
            return uuid in self._transferred_uuids

    def mark_transfer_attempt(self, uuid: str, did: str, target_ip: str) -> bool:
        """
        Mark a transfer attempt for a UUID.
        Returns False if already transferred, True if this is the first attempt.
        """
        with self._lock:
            if uuid in self._transferred_uuids:
                return False

            record = TransferRecord(uuid, did, target_ip)
            record.status = TransferStatus.IN_PROGRESS
            self._transferred_uuids[uuid] = record
            return True

    def mark_transfer_success(self, uuid: str):
        """Mark a transfer as successful"""
        with self._lock:
            if uuid in self._transferred_uuids:
                self._transferred_uuids[uuid].status = TransferStatus.SUCCESS

    def mark_transfer_failed(self, uuid: str, error: str):
        """Mark a transfer as failed"""
        with self._lock:
            if uuid in self._transferred_uuids:
                record = self._transferred_uuids[uuid]
                record.status = TransferStatus.FAILED
                record.error_message = error

    def get_transfer_record(self, uuid: str) -> Optional[TransferRecord]:
        """Get transfer record for a UUID"""
        with self._lock:
            return self._transferred_uuids.get(uuid)

    def increment_attempt(self, uuid: str):
        """Increment attempt counter for a UUID"""
        with self._lock:
            if uuid in self._transferred_uuids:
                self._transferred_uuids[uuid].attempt_count += 1


class FreeSWITCHTransferManager:
    """
    Manages SIP transfers for FreeSWITCH calls using uuid_deflect (SIP REFER).

    This class replicates the pjsua2 ViciDialApiTransfer functionality for FreeSWITCH.
    Key differences:
    - Uses FreeSWITCH ESL API instead of pjsua2 SIP stack
    - Uses uuid_deflect command for SIP REFER
    - Event-driven confirmation via CHANNEL_BRIDGE events
    """

    # Maximum transfer retry attempts
    MAX_RETRIES = 3

    # Base delay between retries (seconds)
    RETRY_DELAY = 0.5

    def __init__(self,
                 qualified_did: str,
                 hangup_did: str,
                 server_ip: str,
                 logger: logging.Logger):
        """
        Initialize the transfer manager.

        Args:
            qualified_did: DID for qualified transfers (from database, optional)
            hangup_did: DID for hangup transfers (from database, optional)
            server_ip: Target IP address (source IP from incoming call)
            logger: Logger instance
        """
        # Store DIDs (allow None/empty for independent operation)
        self.qualified_did = str(qualified_did).strip() if qualified_did else None
        self.hangup_did = str(hangup_did).strip() if hangup_did else None
        self.server_ip = server_ip
        self.logger = logger

        # Thread-safe transfer tracker
        self._tracker = ThreadSafeTransferTracker()

        # Log transfer capabilities
        if self.qualified_did:
            self.logger.info(f"✓ Qualified transfers enabled: DID {self.qualified_did}")
        else:
            self.logger.info("✗ Qualified transfers disabled: no DID configured")

        if self.hangup_did:
            self.logger.info(f"✓ Hangup transfers enabled: DID {self.hangup_did}")
        else:
            self.logger.info("✗ Hangup transfers disabled: no DID configured")

        self.logger.info(f"Transfer Manager ready with server_ip={self.server_ip}")

    def _validate_did(self, did: str) -> bool:
        """
        Check if DID is provided (not None/empty).

        Args:
            did: DID to check

        Returns:
            True if DID is provided, False if None/empty
        """
        return bool(did and did.strip())

    def _build_sip_uri(self, did: str) -> str:
        """
        Build SIP URI for transfer target.

        Format: sip:{did}@{server_ip}

        Args:
            did: Destination DID

        Returns:
            Complete SIP URI string
        """
        return f"sip:{did}@{self.server_ip}"

    def _perform_transfer_with_retries(self,
                                       esl_connection: Any,
                                       uuid: str,
                                       did: str,
                                       operation_name: str,
                                       customer_phone: Optional[str] = None) -> bool:
        """
        Perform transfer with retry logic.

        Args:
            esl_connection: FreeSWITCH ESL connection object
            uuid: Channel UUID
            did: Destination DID
            operation_name: Description of operation (for logging)
            customer_phone: Customer phone number (optional, for logging)

        Returns:
            True if transfer successful, False otherwise
        """
        # Validate DID
        if not self._validate_did(did):
            self.logger.error(f"Transfer aborted: Invalid DID '{did}'")
            return False

        # Build SIP URI
        sip_uri = self._build_sip_uri(did)

        # Check if already transferred
        if self._tracker.is_already_transferred(uuid):
            self.logger.warning(f"Call {uuid} already transferred - skipping duplicate")
            return False

        # Mark transfer attempt
        if not self._tracker.mark_transfer_attempt(uuid, did, self.server_ip):
            self.logger.warning(f"Failed to mark transfer attempt for {uuid}")
            return False

        # Retry loop
        attempt = 0
        while attempt < self.MAX_RETRIES:
            attempt += 1
            self._tracker.increment_attempt(uuid)

            try:
                self.logger.info(
                    f"Attempt {attempt}/{self.MAX_RETRIES}: {operation_name} "
                    f"UUID={uuid} to {sip_uri}"
                    + (f" (customer: {customer_phone})" if customer_phone else "")
                )

                # Execute FreeSWITCH uuid_deflect (SIP REFER)
                result = esl_connection.api("uuid_deflect", f"{uuid} {sip_uri}")

                if result:
                    response_body = result.getBody()

                    # Check for success
                    if "+OK" in response_body or "Success" in response_body:
                        self.logger.info(
                            f"Transfer successful: {operation_name} to {sip_uri} "
                            f"(attempt {attempt})"
                        )
                        self._tracker.mark_transfer_success(uuid)
                        return True
                    else:
                        error_msg = response_body.strip()
                        self.logger.warning(
                            f"Transfer attempt {attempt} failed: {error_msg}"
                        )

                        # Don't retry on certain errors
                        if "Cannot locate session" in error_msg or \
                           "No such channel" in error_msg:
                            self.logger.error(
                                f"Call {uuid} no longer exists - aborting transfer"
                            )
                            self._tracker.mark_transfer_failed(uuid, error_msg)
                            return False
                else:
                    self.logger.warning(f"Transfer attempt {attempt}: No response from FreeSWITCH")

                # Wait before retry (exponential backoff)
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * attempt
                    self.logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)

            except Exception as e:
                error_msg = f"Exception during transfer attempt {attempt}: {e}"
                self.logger.error(error_msg)

                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * attempt
                    time.sleep(delay)
                else:
                    self._tracker.mark_transfer_failed(uuid, error_msg)

        # All retries exhausted
        self.logger.error(
            f"Transfer failed after {self.MAX_RETRIES} attempts: {operation_name}"
        )
        self._tracker.mark_transfer_failed(uuid, "Max retries exceeded")
        return False

    def transfer_qualified_call(self,
                               esl_connection: Any,
                               uuid: str,
                               customer_phone: Optional[str] = None) -> bool:
        """
        Transfer a qualified call to the qualified DID.

        This is equivalent to pjsua2's ViciDialApiTransfer.transfer_qualified_call().
        Used when a lead is qualified and should be transferred to an agent.

        Args:
            esl_connection: FreeSWITCH ESL connection object
            uuid: Channel UUID
            customer_phone: Customer phone number (optional)

        Returns:
            True if transfer successful, False otherwise
        """
        # Check if qualified DID is configured
        if not self.qualified_did:
            self.logger.warning("Qualified transfer skipped: no qualified_did configured")
            return False

        return self._perform_transfer_with_retries(
            esl_connection,
            uuid,
            self.qualified_did,
            "Qualified Transfer",
            customer_phone
        )

    def hangup_call_via_did(self,
                           esl_connection: Any,
                           uuid: str) -> bool:
        """
        Hangup call by transferring to hangup DID.

        This is equivalent to pjsua2's ViciDialApiTransfer.hangup_call_via_did().
        Used for clean call termination via transfer to a hangup extension.

        Args:
            esl_connection: FreeSWITCH ESL connection object
            uuid: Channel UUID

        Returns:
            True if transfer successful, False otherwise
        """
        # Check if hangup DID is configured
        if not self.hangup_did:
            self.logger.info("Hangup transfer skipped: no hangup_did configured")
            return False

        return self._perform_transfer_with_retries(
            esl_connection,
            uuid,
            self.hangup_did,
            "Hangup Transfer"
        )

    def get_transfer_status(self, uuid: str) -> Optional[TransferRecord]:
        """
        Get transfer status for a call UUID.

        Args:
            uuid: Channel UUID

        Returns:
            TransferRecord if exists, None otherwise
        """
        return self._tracker.get_transfer_record(uuid)

    def is_call_transferred(self, uuid: str) -> bool:
        """
        Check if a call has been transferred.

        Args:
            uuid: Channel UUID

        Returns:
            True if transferred, False otherwise
        """
        record = self._tracker.get_transfer_record(uuid)
        return record is not None and record.status == TransferStatus.SUCCESS
