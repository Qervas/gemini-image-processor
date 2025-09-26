#!/usr/bin/env python3
"""
Gemini Sky Removal - Main Entry Point

Professional sky removal using Google Gemini 2.5 Flash API.
Features: PyQt6 GUI, batch processing, rate limiting, gallery view.
"""

import sys
import os
from pathlib import Path

def main():
    """Main application entry point"""
    print("🦍 Gemini Sky Removal")
    print("=" * 30)
    print("Loading GUI...")

    try:
        # Import and run the GUI
        from gui.main_window import main as gui_main
        gui_main()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()