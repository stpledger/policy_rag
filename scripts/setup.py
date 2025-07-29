#!/usr/bin/env python3
"""
Setup Script for Education Policy RAG System

This script initializes the RAG system with proper directory structure,
validates configuration, and prepares the system for operation.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import get_config, print_config

def main():
    """Initialize the RAG system."""
    print("🚀 Education Policy RAG System Setup")
    print("=" * 50)
    
    # Check configuration
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        print_config()
        
        # Validate system
        config.validate(check_files=True)  # Explicitly check files during setup
        print("✅ System validation passed")
        
        print("\n🎉 Setup complete! You can now:")
        print("   • Run the web app: streamlit run app.py")
        print("   • Run validation: python scripts/validate.py")
        print("   • Run tests: python -m pytest tests/")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
