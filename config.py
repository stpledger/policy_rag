"""
Configuration management for the RAG system.
Centralizes all configuration settings with validation and environment variable support.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RAGConfig:
    """Configuration settings for the RAG system."""
    
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
