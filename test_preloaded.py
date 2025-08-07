#!/usr/bin/env python3
"""
Test script to manually set the preloaded flag and verify document processing is skipped.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import the global variables from the run.py module
import app.api.endpoints.run as run_module

def set_preloaded_flag():
    """
    Manually sets the policy_preloaded flag to True to test skipping document processing.
    """
    print("🔧 Setting policy_preloaded flag to True...")
    run_module.policy_preloaded = True
    print("✅ policy_preloaded flag set to True")
    print("📝 Next API request should skip document processing")

if __name__ == "__main__":
    set_preloaded_flag() 