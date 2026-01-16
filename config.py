"""
Configuration file for VideoRAG
"""
import os
from pathlib import Path

# Model paths
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl")

# Paths
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./video_indexes"))
CACHE_SIZE_MB = int(os.getenv("CACHE_SIZE_MB", "1000"))

# Processing defaults
DEFAULT_FPS = float(os.getenv("DEFAULT_FPS", "1.0"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
