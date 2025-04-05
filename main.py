#!/usr/bin/env python3
"""
Main entry point for the Continuous Speaker Verification System.
Run this file to start the application.
"""

import sys
import os
from src.demo_app import main

if __name__ == "__main__":
    # Ensure the current directory is in the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Call the main function from the demo application
    main() 