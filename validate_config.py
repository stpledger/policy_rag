#!/usr/bin/env python3
"""
Configuration validation script for the RAG system.
Run this to check if your configuration is set up correctly.
"""

import sys
import os
from config import get_config, print_config

def validate_environment():
    """Validate the environment and configuration."""
    print("🔍 Validating RAG System Configuration...")
    print("=" * 50)
    
    try:
        # Load configuration
        config = get_config()
        
        # Print current configuration
        print_config()
        print()
        
        # Validate configuration
        print("🔧 Running configuration validation...")
        config.validate()
        print("✅ Configuration validation passed!")
        
        # Check file existence
        print("\n📁 Checking file dependencies...")
        
        if os.path.exists(config.db_path):
            print(f"✅ Database found: {config.db_path}")
        else:
            print(f"⚠️  Database not found: {config.db_path}")
            print("   Run scraping first to create the database")
        
        if os.path.exists(config.vectorstore_path):
            print(f"✅ Vectorstore found: {config.vectorstore_path}")
        else:
            print(f"⚠️  Vectorstore not found: {config.vectorstore_path}")
            print("   Run vectorization first to create the vectorstore")
        
        # Check API key
        print("\n🔑 Checking API credentials...")
        if config.openai_api_key and config.openai_api_key != "your_openai_api_key_here":
            print("✅ OpenAI API key is configured")
        else:
            print("❌ OpenAI API key not configured")
            print("   Set OPENAI_API_KEY in your .env file")
            return False
        
        print("\n🎉 Configuration validation completed successfully!")
        return True
        
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = validate_environment()
    if not success:
        sys.exit(1)
    
    print("\n🚀 RAG system is ready to use!")
    print("   - Run 'streamlit run app.py' to start the web interface")
    print("   - Or import from rag_pipeline in your Python code")
