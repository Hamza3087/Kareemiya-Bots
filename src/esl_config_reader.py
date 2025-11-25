#!/usr/bin/env python3
"""
Dynamic FreeSWITCH ESL Configuration Reader
Reads event_socket.conf.xml at runtime (no hardcoded credentials)
"""

import xml.etree.ElementTree as ET
import logging
import threading
from typing import Optional, Dict
import os


class ESLConfigReader:
    """
    Reads ESL configuration from FreeSWITCH config file.
    Caches result to avoid repeated file I/O.
    Thread-safe singleton pattern.
    """

    _cached_config: Optional[Dict[str, str]] = None
    _cache_lock = threading.RLock()

    DEFAULT_CONFIG_PATH = "/usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml"

    # Fallback values (FreeSWITCH defaults)
    FALLBACK_HOST = "127.0.0.1"
    FALLBACK_PORT = "8021"
    FALLBACK_PASSWORD = "ClueCon"

    @classmethod
    def get_esl_config(cls, config_path: Optional[str] = None) -> Dict[str, str]:
        """
        Get ESL configuration (cached after first read).

        Args:
            config_path: Optional custom path to event_socket.conf.xml

        Returns:
            Dict with keys: 'host', 'port', 'password'

        Example:
            >>> config = ESLConfigReader.get_esl_config()
            >>> print(config)
            {'host': '127.0.0.1', 'port': '8021', 'password': 'lS9PUmz8...'}
        """
        with cls._cache_lock:
            if cls._cached_config is not None:
                return cls._cached_config

            cls._cached_config = cls._read_config_from_file(
                config_path or cls.DEFAULT_CONFIG_PATH
            )
            return cls._cached_config

    @classmethod
    def _read_config_from_file(cls, config_path: str) -> Dict[str, str]:
        """
        Parse XML config file and extract ESL parameters.
        Falls back to defaults if parsing fails.

        Args:
            config_path: Path to event_socket.conf.xml

        Returns:
            Dict with ESL connection parameters
        """
        logger = logging.getLogger(__name__)

        try:
            if not os.path.exists(config_path):
                logger.warning(f"ESL config not found: {config_path}, using defaults")
                return cls._get_fallback_config()

            tree = ET.parse(config_path)
            root = tree.getroot()

            config = {
                'host': cls.FALLBACK_HOST,
                'port': cls.FALLBACK_PORT,
                'password': cls.FALLBACK_PASSWORD
            }

            # Extract parameters from XML
            # Format: <param name="listen-ip" value="::"/>
            for param in root.findall('.//param'):
                name = param.get('name')
                value = param.get('value')

                if not name or not value:
                    continue

                if name == 'listen-ip':
                    # Convert :: (IPv6 all) or 0.0.0.0 to 127.0.0.1 for localhost connection
                    # FreeSWITCH binds to all interfaces, but we connect via localhost
                    if value in ['::', '0.0.0.0']:
                        config['host'] = '127.0.0.1'
                    else:
                        config['host'] = value

                elif name == 'listen-port':
                    config['port'] = value

                elif name == 'password':
                    config['password'] = value

            logger.info(f"✓ ESL config loaded from {config_path}: {config['host']}:{config['port']}")
            return config

        except ET.ParseError as e:
            logger.error(f"XML parse error in {config_path}: {e}, using defaults")
            return cls._get_fallback_config()

        except Exception as e:
            logger.error(f"Failed to read ESL config from {config_path}: {e}, using defaults")
            return cls._get_fallback_config()

    @classmethod
    def _get_fallback_config(cls) -> Dict[str, str]:
        """
        Return default FreeSWITCH ESL configuration.
        Used when config file is missing or unreadable.
        """
        logger = logging.getLogger(__name__)
        logger.warning(f"Using fallback ESL config: {cls.FALLBACK_HOST}:{cls.FALLBACK_PORT}")

        return {
            'host': cls.FALLBACK_HOST,
            'port': cls.FALLBACK_PORT,
            'password': cls.FALLBACK_PASSWORD
        }

    @classmethod
    def clear_cache(cls):
        """
        Clear cached config (for testing or config reload).

        Example:
            >>> ESLConfigReader.clear_cache()
            >>> config = ESLConfigReader.get_esl_config()  # Re-reads from file
        """
        with cls._cache_lock:
            cls._cached_config = None

    @classmethod
    def validate_connection(cls, config: Optional[Dict[str, str]] = None) -> bool:
        """
        Validate ESL connection with given or cached config.
        Useful for health checks and testing.

        Args:
            config: Optional config dict, uses cached config if None

        Returns:
            True if connection successful, False otherwise

        Note:
            Requires ESL module to be imported
        """
        if config is None:
            config = cls.get_esl_config()

        logger = logging.getLogger(__name__)

        try:
            import ESL

            conn = ESL.ESLconnection(
                config['host'],
                config['port'],
                config['password']
            )

            if conn.connected():
                conn.disconnect()
                logger.info(f"✓ ESL connection validated: {config['host']}:{config['port']}")
                return True
            else:
                logger.error(f"✗ ESL connection failed: {config['host']}:{config['port']}")
                return False

        except ImportError:
            logger.error("ESL module not available for validation")
            return False

        except Exception as e:
            logger.error(f"ESL connection validation failed: {e}")
            return False
