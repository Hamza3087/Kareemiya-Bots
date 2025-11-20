#!/usr/bin/env python3
"""
File Event Watcher using inotify for efficient file monitoring

Replaces inefficient polling (time.sleep) with kernel-level event notifications.
Provides automatic fallback to polling if inotify is unavailable.
"""

import os
import time
import logging
import threading
from typing import Optional

# Try to import inotify_simple, but gracefully degrade if not available
try:
    from inotify_simple import INotify, flags
    INOTIFY_AVAILABLE = True
except ImportError:
    INOTIFY_AVAILABLE = False
    INotify = None
    flags = None


class FileEventWatcher:
    """
    Efficient file modification watcher using Linux inotify.
    Falls back to polling if inotify is unavailable.

    Thread-safe for single file monitoring.
    """

    def __init__(self, file_path: str, logger=None, use_inotify: bool = True,
                 fallback_interval: float = 0.5):
        """
        Initialize file watcher for a specific file.

        Args:
            file_path: Absolute path to file to watch
            logger: Logger instance (optional)
            use_inotify: Whether to try using inotify (True) or force polling (False)
            fallback_interval: Polling interval in seconds if inotify unavailable
        """
        self.file_path = file_path
        self.logger = logger or logging.getLogger(__name__)
        self.fallback_interval = fallback_interval

        # State
        self.inotify = None
        self.watch_descriptor = None
        self.using_inotify = False
        self.last_poll_time = 0
        self.lock = threading.RLock()

        # Try to initialize inotify if requested and available
        if use_inotify and INOTIFY_AVAILABLE:
            try:
                self._init_inotify()
            except Exception as e:
                self.logger.warning(f"Failed to initialize inotify for {file_path}: {e}")

    def _init_inotify(self):
        """Initialize inotify watching for the file"""
        with self.lock:
            # Create inotify instance
            self.inotify = INotify()

            # Get directory and filename
            directory = os.path.dirname(self.file_path)

            # Watch the directory (not the file directly, as file might not exist yet)
            # We'll filter for our specific file in the event handler
            try:
                self.watch_descriptor = self.inotify.add_watch(
                    directory,
                    flags.MODIFY | flags.CREATE | flags.DELETE | flags.MOVED_TO
                )
                self.using_inotify = True
            except Exception as e:
                self.logger.warning(f"Failed to add inotify watch: {e}")
                self.inotify = None
                raise

    def wait_for_modification(self, timeout_ms: int = 2000) -> bool:
        """
        Wait for file modification event.

        Args:
            timeout_ms: Maximum time to wait in milliseconds (0 = non-blocking)

        Returns:
            True if file was modified, False if timeout occurred
        """
        if self.using_inotify and self.inotify:
            return self._wait_inotify(timeout_ms)
        else:
            return self._wait_polling(timeout_ms)

    def _wait_inotify(self, timeout_ms: int) -> bool:
        """Wait using inotify events"""
        try:
            filename = os.path.basename(self.file_path)

            # Read events with timeout (convert ms to seconds)
            timeout_sec = timeout_ms / 1000.0 if timeout_ms > 0 else 0

            events = self.inotify.read(timeout=int(timeout_ms))

            # Check if any event is for our file
            for event in events:
                if event.name == filename:
                    # File was modified/created/moved
                    return True

            # Events received but not for our file, or timeout occurred
            return False

        except Exception as e:
            self.logger.warning(f"inotify error: {e}, falling back to polling")
            # Disable inotify and fall back to polling
            self._disable_inotify()
            return self._wait_polling(timeout_ms)

    def _wait_polling(self, timeout_ms: int) -> bool:
        """Fallback: Wait using polling with sleep"""
        # Convert timeout to seconds
        timeout_sec = timeout_ms / 1000.0 if timeout_ms > 0 else 0

        # For polling, we just sleep for the minimum of timeout or fallback_interval
        sleep_time = min(timeout_sec, self.fallback_interval) if timeout_sec > 0 else self.fallback_interval

        if sleep_time > 0:
            time.sleep(sleep_time)

        # In polling mode, we always return True to trigger a check
        # (The caller will check file size to see if there's actually new data)
        return True

    def _disable_inotify(self):
        """Disable inotify and switch to polling mode"""
        with self.lock:
            if self.inotify:
                try:
                    self.inotify.close()
                except:
                    pass
                self.inotify = None
            self.watch_descriptor = None
            self.using_inotify = False

    def close(self):
        """Clean up inotify resources"""
        with self.lock:
            if self.inotify:
                try:
                    self.inotify.close()
                except Exception as e:
                    pass
                finally:
                    self.inotify = None
            self.watch_descriptor = None
            self.using_inotify = False

    def is_using_inotify(self) -> bool:
        """Check if currently using inotify (vs polling)"""
        return self.using_inotify

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources"""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure cleanup"""
        self.close()
