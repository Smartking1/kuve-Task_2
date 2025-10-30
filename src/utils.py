import re
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Overlap between chunks (in characters)
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    # Approximate words per chunk (assuming ~5 chars per word)
    words_per_chunk = chunk_size // 5
    overlap_words = overlap // 5
    
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk = ' '.join(words[i:i + words_per_chunk])
        if chunk:
            chunks.append(chunk)
        
        # Break if we've processed all words
        if i + words_per_chunk >= len(words):
            break
    
    return chunks


def save_json(data: Any, filepath: Path) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: Path) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def format_document_info(doc_data: Dict[str, Any]) -> str:
    """
    Format document information for display.
    
    Args:
        doc_data: Dictionary containing document metadata
        
    Returns:
        Formatted string
    """
    info = []
    info.append(f"Document: {doc_data.get('filename', 'Unknown')}")
    info.append(f"Pages: {doc_data.get('num_pages', 'N/A')}")
    info.append(f"Chunks: {doc_data.get('num_chunks', 'N/A')}")
    info.append(f"Characters: {doc_data.get('num_chars', 'N/A')}")
    
    return '\n'.join(info)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_file_size(filepath: Path) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
        
    Returns:
        Formatted file size string
    """
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"