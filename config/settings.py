import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data subdirectories
DOCUMENTS_DIR = DATA_DIR / "documents"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUMMARIES_DIR = DATA_DIR / "summaries"

# Model paths
EMBEDDING_MODEL_DIR = MODELS_DIR / "embeddings"
SUMMARIZATION_MODEL_DIR = MODELS_DIR / "summarization"

# Create directories if they don't exist
for directory in [DATA_DIR, DOCUMENTS_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR,
                  MODELS_DIR, EMBEDDING_MODEL_DIR, SUMMARIZATION_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use TinyLlama for both summarization and Q&A (lightweight, fast)
TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Processing parameters
MAX_CHUNK_SIZE = 768  # Increased from 512 for better context
CHUNK_OVERLAP = 100    # Increased from 50 for better continuity
TOP_K_RETRIEVAL = 5   # Increased from 3 for better answers

# Summarization parameters
MAX_SUMMARY_LENGTH = 250
MIN_SUMMARY_LENGTH = 50

# Q&A parameters
MAX_ANSWER_LENGTH = 300  # Allow longer, more detailed answers
USE_CONTEXT_WINDOW = True  # Expand chunks with surrounding context

# TinyLlama generation parameters
TINYLLAMA_MAX_NEW_TOKENS = 256
TINYLLAMA_TEMPERATURE = 0.7
TINYLLAMA_TOP_K = 50
TINYLLAMA_TOP_P = 0.95

# FAISS parameters
FAISS_INDEX_TYPE = "Flat"  # Can be "Flat" or "IVF" for large datasets