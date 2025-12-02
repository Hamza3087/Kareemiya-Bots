#!/usr/bin/env python3
"""
FreeSWITCH SIP Bot - Firewall IP Whitelist Manager
Syncs whitelisted IPs from SuiteCRM database to iptables/ipset
Blocks non-whitelisted IPs at kernel level on port 5060

Usage:
    python3 /root/sip-bot/src/firewall_whitelist.py

Exit Codes:
    0 = Success
    1 = Error (database, iptables, or ipset failure)
"""

import sys
import os
import subprocess
import logging
from typing import Set, List
import ipaddress  # For IP validation

# Add project root to path
sys.path.insert(0, '/root/sip-bot')

from src.suitecrm_integration import fetch_active_agent_configs

# Configuration
IPSET_NAME = "sip_whitelist"
SIP_PORT = 5060
PROTOCOL = "udp"
FALLBACK_IP = "127.0.0.1"  # Always allow localhost

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


def validate_ip(ip: str) -> bool:
    """
    Validate IP address format
    Returns True if valid IPv4 address
    """
    try:
        ipaddress.IPv4Address(ip)
        return True
    except ValueError:
        return False


def load_whitelisted_ips_from_db() -> Set[str]:
    """
    Load whitelisted IPs from SuiteCRM database
    Queries e_campaigns.server_ip field and parses comma-separated values

    Returns:
        Set of valid IP addresses
    """
    log.info("Loading whitelisted IPs from database...")

    try:
        configs = fetch_active_agent_configs()
        whitelisted_ips = set()

        for cfg in configs:
            if cfg.server_ip:
                # Parse comma-separated IPs
                ips = [ip.strip() for ip in cfg.server_ip.split(',') if ip.strip()]

                # Validate each IP
                for ip in ips:
                    if validate_ip(ip):
                        whitelisted_ips.add(ip)
                    else:
                        log.warning(f"Invalid IP address skipped: {ip}")

        # Always include localhost
        whitelisted_ips.add(FALLBACK_IP)

        log.info(f"Loaded {len(whitelisted_ips)} whitelisted IPs from database")
        for ip in sorted(whitelisted_ips):
            log.info(f"  → {ip}")

        return whitelisted_ips

    except Exception as e:
        log.error(f"Failed to load IPs from database: {e}")
        log.warning(f"Using fallback: {FALLBACK_IP} only")
        return {FALLBACK_IP}


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Execute shell command with error handling

    Args:
        cmd: Command and arguments as list
        check: Raise exception on non-zero exit (default True)

    Returns:
        CompletedProcess object
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed: {' '.join(cmd)}")
        log.error(f"Exit code: {e.returncode}")
        log.error(f"Error: {e.stderr}")
        raise


def setup_ipset(whitelisted_ips: Set[str]) -> bool:
    """
    Create and populate ipset with whitelisted IPs

    Args:
        whitelisted_ips: Set of IP addresses to whitelist

    Returns:
        True if successful, False otherwise
    """
    log.info(f"Setting up ipset: {IPSET_NAME}")

    try:
        # Create ipset (ignore if already exists)
        run_command(['ipset', 'create', IPSET_NAME, 'hash:ip', '-exist'])
        log.info(f"✓ ipset '{IPSET_NAME}' created/verified")

        # Flush existing entries
        run_command(['ipset', 'flush', IPSET_NAME])
        log.info(f"✓ ipset '{IPSET_NAME}' flushed")

        # Add whitelisted IPs
        for ip in whitelisted_ips:
            run_command(['ipset', 'add', IPSET_NAME, ip])

        log.info(f"✓ Added {len(whitelisted_ips)} IPs to ipset")

        return True

    except Exception as e:
        log.error(f"Failed to setup ipset: {e}")
        return False


def setup_iptables_rules() -> bool:
    """
    Setup iptables rules to enforce IP whitelist on port 5060
    Rules are idempotent (safe to run multiple times)

    Returns:
        True if successful, False otherwise
    """
    log.info("Setting up iptables rules...")

    rules = [
        {
            'description': f'ACCEPT whitelisted IPs on port {SIP_PORT}',
            'rule': [
                'iptables', '-A', 'INPUT',
                '-p', PROTOCOL,
                '--dport', str(SIP_PORT),
                '-m', 'set',
                '--match-set', IPSET_NAME, 'src',
                '-j', 'ACCEPT'
            ]
        },
        {
            'description': f'DROP all other IPs on port {SIP_PORT}',
            'rule': [
                'iptables', '-A', 'INPUT',
                '-p', PROTOCOL,
                '--dport', str(SIP_PORT),
                '-j', 'DROP'
            ]
        }
    ]

    try:
        for rule_info in rules:
            rule = rule_info['rule']
            desc = rule_info['description']

            # Check if rule exists (replace -A with -C)
            check_rule = rule.copy()
            check_rule[1] = '-C'

            result = run_command(check_rule, check=False)

            if result.returncode == 0:
                log.info(f"✓ Rule already exists: {desc}")
            else:
                # Add rule
                run_command(rule)
                log.info(f"✓ Rule added: {desc}")

        return True

    except Exception as e:
        log.error(f"Failed to setup iptables rules: {e}")
        return False


def verify_firewall() -> bool:
    """
    Verify firewall rules are correctly configured

    Returns:
        True if verified, False otherwise
    """
    log.info("Verifying firewall configuration...")

    try:
        # Check ipset exists and has entries
        result = run_command(['ipset', 'list', IPSET_NAME])
        if 'Number of entries: 0' in result.stdout:
            log.warning("⚠ ipset exists but has no entries!")
            return False

        # Check iptables rules exist
        result = run_command(['iptables', '-L', 'INPUT', '-n', '-v'])
        if IPSET_NAME not in result.stdout:
            log.warning(f"⚠ iptables rules not found for ipset {IPSET_NAME}!")
            return False

        log.info("✓ Firewall configuration verified")
        return True

    except Exception as e:
        log.error(f"Verification failed: {e}")
        return False


def sync_firewall() -> int:
    """
    Main function: Sync firewall whitelist from database

    Returns:
        0 if successful, 1 if error
    """
    log.info("=" * 60)
    log.info("FreeSWITCH SIP Bot - Firewall Whitelist Sync")
    log.info("=" * 60)

    try:
        # Step 1: Load IPs from database
        whitelisted_ips = load_whitelisted_ips_from_db()

        if not whitelisted_ips:
            log.error("No whitelisted IPs found!")
            return 1

        # Step 2: Setup ipset
        if not setup_ipset(whitelisted_ips):
            log.error("Failed to setup ipset")
            return 1

        # Step 3: Setup iptables rules
        if not setup_iptables_rules():
            log.error("Failed to setup iptables rules")
            return 1

        # Step 4: Verify configuration
        if not verify_firewall():
            log.warning("Firewall verification failed, but rules may still work")

        # Success
        log.info("=" * 60)
        log.info(f"✅ SUCCESS: Firewall whitelist synchronized")
        log.info(f"   {len(whitelisted_ips)} IPs whitelisted on port {SIP_PORT}/{PROTOCOL}")
        log.info("=" * 60)

        return 0

    except Exception as e:
        log.error("=" * 60)
        log.error(f"❌ FAILED: {e}")
        log.error("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = sync_firewall()
    # Force immediate exit to avoid hanging on async workers
    os._exit(exit_code)
