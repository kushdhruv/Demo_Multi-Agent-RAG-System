#!/usr/bin/env python3
"""
Script to manually set the preloaded state in the service manager.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import the service manager
from app.api.endpoints.run import service_manager

def set_preloaded_state():
    """
    Manually sets the policy_preloaded flag to True.
    """
    print("🔧 Setting policy_preloaded flag to True...")
    service_manager.policy_preloaded = True
    print("✅ policy_preloaded flag set to True")
    print("📝 Next API request should skip document processing")
    print(f"🔍 Current state: policy_preloaded = {service_manager.policy_preloaded}")

if __name__ == "__main__":
    set_preloaded_state() 