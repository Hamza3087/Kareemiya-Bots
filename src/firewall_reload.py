#!/usr/bin/env python3
"""
Manual Firewall Reload Script
Reloads IP whitelist without restarting sip_bot.service

Usage:
    sudo python3 /root/sip-bot/src/firewall_reload.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/root/sip-bot')

from src.firewall_whitelist import sync_firewall

if __name__ == "__main__":
    print("\n🔄 Manually reloading firewall whitelist from database...\n")
    sys.exit(sync_firewall())
