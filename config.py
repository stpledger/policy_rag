"""
Configuration Management for Education Policy RAG System
======================================================

This module provides centralized configuration management for the RAG system,
supporting environment variables, validation, and default settings.

Key Features:
- Environment variable integration with fallback defaults
- Comprehensive validation of all configuration parameters
- Type-safe configuration using dataclasses
- Support for different deployment environments

Classes:
    RAGConfig: Main configuration dataclass with all system settings

Functions:
    get_config(): Get the global configuration instance
    reload_config(): Reload configuration from environment
    print_config(): Display current configuration for debugging

Example Usage:
    >>> from config import get_config
    >>> config = get_config()
    >>> print(config.model_name)
    gpt-4
    
    # Override with environment variables
    >>> import os
    >>> os.environ['RAG_MODEL_NAME'] = 'gpt-3.5-turbo'
    >>> config = reload_config()
    >>> print(config.model_name)
    gpt-3.5-turbo

Environment Variables:
    Required:
        OPENAI_API_KEY: OpenAI API key for language models and embeddings
        
    Database & Storage:
        RAG_DB_PATH: Path to SQLite database file (default: main.db)
        RAG_VECTORSTORE_PATH: Path to FAISS vectorstore directory (default: ed_policy_vec)
        
    Language Model Settings:
        RAG_MODEL_NAME: OpenAI model for text generation (default: gpt-4)
        RAG_TEMPERATURE: Model creativity/randomness, 0.0-2.0 (default: 0.0)
        RAG_MAX_TOKENS: Maximum tokens in model response (default: 2000)
        
    Retrieval Configuration:
        RAG_RETRIEVAL_K: Number of documents to retrieve (default: 5)
        RAG_SIMILARITY_THRESHOLD: Minimum similarity score, 0.0-1.0 (default: 0.7)
        RAG_CHUNK_SIZE: Size of text chunks for processing (default: 1000)
        RAG_CHUNK_OVERLAP: Overlap between text chunks (default: 200)
        
    Embedding Settings:
        RAG_EMBEDDING_MODEL: OpenAI embedding model (default: text-embedding-ada-002)
        
    Application Settings:
        RAG_APP_TITLE: Web application title (default: 🏛️ Education Policy RAG System)
        RAG_MAX_CHAT_HISTORY: Maximum chat messages to retain (default: 10)
        RAG_ENABLE_ANALYTICS: Enable usage analytics, true/false (default: true)
        
    Performance & Caching:
        RAG_ENABLE_CACHING: Enable response caching, true/false (default: true)
        RAG_CACHE_TTL: Cache time-to-live in seconds (default: 3600)
    
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RAGConfig:
    """
    Configuration settings for the RAG system.
    
    This dataclass contains all configuration parameters for the education policy
    RAG system, including database settings, LLM parameters, retrieval settings,
    and application configurations.
    
    Attributes:
        Database Settings:
            db_path (str): Path to SQLite database file
            vectorstore_path (str): Path to FAISS vector store directory
            
        LLM Settings:
            model_name (str): OpenAI model name for text generation
            temperature (float): Model temperature (0.0-2.0)
            max_tokens (int): Maximum tokens in model response
            
        Retrieval Settings:
            retrieval_k (int): Number of documents to retrieve
            similarity_threshold (float): Minimum similarity score
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
            
        Embedding Settings:
            embedding_model (str): OpenAI embedding model name
            
        API Settings:
            openai_api_key (str): OpenAI API key
            
        Application Settings:
            app_title (str): Application title for web interface
            max_chat_history (int): Maximum chat history to maintain
            enable_analytics (bool): Whether to enable analytics
            
        Performance Settings:
            enable_caching (bool): Whether to enable response caching
            cache_ttl (int): Cache time-to-live in seconds
    
    Methods:
        validate(): Validate all configuration parameters
        from_env(): Create configuration from environment variables
        to_dict(): Convert configuration to dictionary
    
    Example:
        >>> config = RAGConfig()
        >>> config.validate()
        >>> print(config.model_name)
        gpt-4
    """
    
    # Database settings
    db_path: str = "main.db"
    vectorstore_path: str = "ed_policy_vec"
    
    # LLM settings
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    
    # Retrieval settings
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding settings
    embedding_model: str = "text-embedding-ada-002"
    
    # API keys and authentication
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Application settings
    app_title: str = "🏛️ Education Policy RAG System"
    max_chat_history: int = 10
    enable_analytics: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate API keys
        if not self.openai_api_key:
            errors.append("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Validate numeric values
        if self.retrieval_k <= 0:
            errors.append("retrieval_k must be positive")
        
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            errors.append("chunk_overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        # Validate file paths
        if not os.path.exists(self.db_path):
            errors.append(f"Database file not found: {self.db_path}")
        
        if not os.path.exists(self.vectorstore_path):
            errors.append(f"Vectorstore directory not found: {self.vectorstore_path}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables with defaults."""
        return cls(
            # Database settings
            db_path=os.getenv("RAG_DB_PATH", cls.db_path),
            vectorstore_path=os.getenv("RAG_VECTORSTORE_PATH", cls.vectorstore_path),

            # LLM settings
            model_name=os.getenv("RAG_MODEL_NAME", cls.model_name),
            temperature=float(os.getenv("RAG_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", cls.max_tokens)),

            # Retrieval settings
            retrieval_k=int(os.getenv("RAG_RETRIEVAL_K", cls.retrieval_k)),
            similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", cls.similarity_threshold)),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),

            # Embedding settings
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),

            # Application settings
            app_title=os.getenv("RAG_APP_TITLE", cls.app_title),
            max_chat_history=int(os.getenv("RAG_MAX_CHAT_HISTORY", cls.max_chat_history)),
            enable_analytics=os.getenv("RAG_ENABLE_ANALYTICS", str(cls.enable_analytics)).lower() == "true",

            # Performance settings
            enable_caching=os.getenv("RAG_ENABLE_CACHING", str(cls.enable_caching)).lower() == "true",
            cache_ttl=int(os.getenv("RAG_CACHE_TTL", cls.cache_ttl)),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging/debugging."""
        config_dict = {
            "db_path": self.db_path,
            "vectorstore_path": self.vectorstore_path,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retrieval_k": self.retrieval_k,
            "similarity_threshold": self.similarity_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "app_title": self.app_title,
            "max_chat_history": self.max_chat_history,
            "enable_analytics": self.enable_analytics,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            # Don't include API keys in logs
            "openai_api_key": "***" if self.openai_api_key else None,
        }
        return config_dict


# Global configuration instance
try:
    config = RAGConfig.from_env()
    config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Using default configuration without validation...")
    config = RAGConfig()


# Utility functions for configuration
def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> RAGConfig:
    """Reload configuration from environment variables."""
    global config
    config = RAGConfig.from_env()
    config.validate()
    return config


def print_config() -> None:
    """Print current configuration (for debugging)."""
    import json
    print("Current RAG System Configuration:")
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    # For testing configuration
    print_config()
