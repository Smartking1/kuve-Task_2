import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pickle
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIR


class EmbeddingsManager:
    """Manage document embeddings for semantic search."""
    
    def __init__(self, use_local: bool = True):
        """
        Initialize the embeddings manager.
        
        Args:
            use_local: Use locally saved model if available
        """
        self.model = None
        self.use_local = use_local
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            if self.use_local and EMBEDDING_MODEL_DIR.exists():
                # Load from local directory
                print(f"Loading local embedding model from {EMBEDDING_MODEL_DIR}")
                self.model = SentenceTransformer(str(EMBEDDING_MODEL_DIR))
            else:
                # Download from HuggingFace
                print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading embedding model: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        return self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def create_document_embeddings(self, documents: List[Dict]) -> Dict:
        """
        Create embeddings for all chunks in documents.
        
        Args:
            documents: List of document data dictionaries
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        all_chunks = []
        chunk_metadata = []
        
        # Collect all chunks with metadata
        for doc in documents:
            for i, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_filename': doc['filename'],
                    'doc_filepath': doc['filepath'],
                    'chunk_index': i,
                    'chunk_text': chunk
                })
        
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self.encode_batch(all_chunks)
        
        return {
            'embeddings': embeddings,
            'metadata': chunk_metadata,
            'num_chunks': len(all_chunks)
        }
    
    def save_embeddings(self, embeddings_data: Dict, filepath: Path):
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            filepath: Path to save file
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings_data, f)
            print(f"✓ Embeddings saved to {filepath}")
        except Exception as e:
            print(f"✗ Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: Path) -> Optional[Dict]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Dictionary containing embeddings and metadata or None
        """
        try:
            with open(filepath, 'rb') as f:
                embeddings_data = pickle.load(f)
            print(f"✓ Embeddings loaded from {filepath}")
            return embeddings_data
        except Exception as e:
            print(f"✗ Error loading embeddings: {e}")
            return None
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embedding vectors.
        
        Returns:
            Embedding dimension
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))