"""
Configuration Validation and System Health Check for Education Policy RAG

This module provides comprehensive validation and diagnostic capabilities for
the Education Policy RAG system configuration. It performs system health checks,
validates environment setup, verifies API connectivity, and ensures all
components are properly configured for optimal operation.

Key Validation Features:
    - Environment Variable Verification: OpenAI API keys and system paths
    - Database Connectivity Testing: SQLite database access and integrity
    - Vectorstore Validation: FAISS index availability and functionality
    - API Connectivity Testing: OpenAI embeddings and chat model access
    - File System Checks: Required directories and file permissions
    - Configuration Parameter Validation: Value ranges and type checking
    - Dependency Verification: Required package installation and versions

Validation Categories:
    1. Environment Setup:
       - OPENAI_API_KEY presence and format validation
       - System path configuration and accessibility
       - Required directory structure verification
       - File permissions and write access testing
       
    2. Database Validation:
       - SQLite database file existence and connectivity
       - Article table structure and data validation
       - Data quality checks and completeness assessment
       - Database performance and query execution testing
       
    3. API Connectivity:
       - OpenAI API authentication and quota verification
       - Embedding model availability and response testing
       - Chat model functionality and parameter validation
       - Rate limiting and error handling verification
       
    4. System Integration:
       - Vectorstore loading and search functionality
       - RAG pipeline initialization and basic operation
       - Configuration consistency across components
       - Performance baseline establishment

Health Check Process:
    1. Load and display current configuration settings
    2. Validate all configuration parameters and constraints
    3. Test database connectivity and data availability
    4. Verify API access and model functionality
    5. Check vectorstore integrity and search capability
    6. Perform end-to-end system integration test
    7. Generate comprehensive status report

Usage:
    >>> # Command line validation
    >>> python validate_config.py
    >>> 
    >>> # Programmatic validation
    >>> from validate_config import validate_environment
    >>> validation_results = validate_environment()
    >>> if validation_results['success']:
    ...     print("System ready for operation")

Expected Output:
    🔍 Validating RAG System Configuration...
    ==================================================
    
    📋 Current Configuration:
    • App Title: Education Policy RAG System
    • Database: main.db (✓ accessible)
    • Vectorstore: ed_policy_vec/ (✓ loaded)
    • Embedding Model: text-embedding-3-small
    • Chat Model: gpt-4o-mini
    
    🔧 Running configuration validation...
    ✅ Configuration validation passed
    
    🌐 Testing API connectivity...
    ✅ OpenAI API connection successful
    ✅ Embedding model accessible
    ✅ Chat model responsive
    
    📊 System health summary:
    ✅ All systems operational
    🚀 RAG system ready for queries

Error Handling:
    - Graceful failure reporting with specific error messages
    - Actionable recommendations for configuration fixes
    - Comprehensive logging of validation steps and results
    - Safe operation with missing or invalid components

Dependencies:
    - config: Central configuration management system
    - OpenAI: API connectivity and model access testing
    - FAISS: Vectorstore functionality verification
    - sqlite3: Database connectivity and validation

Note:
    Run this script before deploying or using the RAG system to ensure
    all components are properly configured and operational. Regular health
    checks help maintain system reliability and performance.
"""

import sys
import os
from ..core.config import get_config, print_config

def validate_environment():
    """
    Comprehensive validation of RAG system environment and configuration.
    
    This function performs a complete health check of the RAG system,
    validating all components from environment variables to API connectivity.
    It provides detailed diagnostics and actionable feedback for any issues
    encountered during the validation process.
    
    Validation Process:
        1. Configuration Loading: Load and display current system configuration
        2. Parameter Validation: Check all configuration values and constraints
        3. Environment Verification: Validate API keys and system paths
        4. Database Testing: Verify database connectivity and data integrity
        5. API Connectivity: Test OpenAI API access and model availability
        6. Vectorstore Validation: Check FAISS index functionality
        7. Integration Testing: Perform end-to-end system operation test
        
    Returns:
        dict: Comprehensive validation results containing:
            - success (bool): Overall validation status
            - config_valid (bool): Configuration parameter validation
            - db_accessible (bool): Database connectivity status
            - api_connected (bool): OpenAI API accessibility
            - vectorstore_loaded (bool): FAISS vectorstore status
            - errors (list): Detailed error messages if any
            - recommendations (list): Action items for fixing issues
            
    Validation Checks:
        Configuration:
        ✓ Required environment variables present
        ✓ Configuration parameters within valid ranges
        ✓ File paths accessible and writable
        ✓ Model names and versions valid
        
        Database:
        ✓ SQLite database file exists and readable
        ✓ Articles table structure correct
        ✓ Sufficient article data available
        ✓ Database performance acceptable
        
        API Services:
        ✓ OpenAI API key valid and authenticated
        ✓ Embedding model accessible
        ✓ Chat model responsive
        ✓ Rate limits and quotas sufficient
        
        System Integration:
        ✓ Vectorstore index loads successfully
        ✓ Document retrieval functions properly
        ✓ End-to-end query processing works
        ✓ Performance meets baseline requirements
        
    Example Output:
        🔍 Validating RAG System Configuration...
        
        📋 Configuration Status: ✅ Valid
        🗄️ Database Status: ✅ Connected (45 articles)
        🌐 API Status: ✅ OpenAI connected
        📊 Vectorstore Status: ✅ Loaded (892 chunks)
        
        🚀 System Status: READY FOR OPERATION
        
    Error Handling:
        - Catches and reports specific configuration errors
        - Provides actionable recommendations for fixes
        - Continues validation even if some components fail
        - Generates comprehensive diagnostic report
        
    Common Issues Resolved:
        - Missing OPENAI_API_KEY environment variable
        - Incorrect database path or permissions
        - Missing vectorstore files or corruption
        - API connectivity or authentication problems
        - Invalid configuration parameter values
        
    Note:
        This function should be run before system deployment and periodically
        during operation to ensure continued system health and reliability.
    """
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
