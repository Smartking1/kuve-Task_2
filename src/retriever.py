import numpy as np
from typing import List, Dict, Tuple
import faiss
from .embeddings_manager import EmbeddingsManager
from config.settings import TOP_K_RETRIEVAL


class DocumentRetriever:
    """Retrieve relevant document chunks using semantic search."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        """
        Initialize the retriever.
        
        Args:
            embeddings_manager: EmbeddingsManager instance
        """
        self.embeddings_manager = embeddings_manager
        self.index = None
        self.metadata = None
        self.embeddings = None
    
    def build_index(self, embeddings_data: Dict):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
        """
        self.embeddings = embeddings_data['embeddings']
        self.metadata = embeddings_data['metadata']
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(self.embeddings)
        
        print(f"✓ FAISS index built with {len(self.embeddings)} vectors")
    
    def search(
        self, 
        query: str, 
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embeddings_manager.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                result = {
                    'chunk_text': self.metadata[idx]['chunk_text'],
                    'doc_filename': self.metadata[idx]['doc_filename'],
                    'doc_filepath': self.metadata[idx]['doc_filepath'],
                    'chunk_index': self.metadata[idx]['chunk_index'],
                    'score': float(score)
                }
                results.append(result)
        
        return results
    
    def search_by_document(
        self, 
        query: str, 
        doc_filename: str, 
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """
        Search within a specific document.
        
        Args:
            query: Search query
            doc_filename: Filename to search within
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        # Get all results
        all_results = self.search(query, top_k=len(self.metadata))
        
        # Filter by document
        filtered_results = [
            r for r in all_results 
            if r['doc_filename'] == doc_filename
        ]
        
        return filtered_results[:top_k]
    
    def get_context_window(
        self, 
        chunk_index: int, 
        doc_filename: str, 
        window_size: int = 1
    ) -> str:
        """
        Get surrounding context for a chunk.
        
        Args:
            chunk_index: Index of the chunk
            doc_filename: Document filename
            window_size: Number of chunks before/after to include
            
        Returns:
            Combined context text
        """
        # Find all chunks from the document
        doc_chunks = [
            m for m in self.metadata 
            if m['doc_filename'] == doc_filename
        ]
        
        # Sort by chunk index
        doc_chunks.sort(key=lambda x: x['chunk_index'])
        
        # Find the target chunk position
        target_pos = None
        for i, chunk in enumerate(doc_chunks):
            if chunk['chunk_index'] == chunk_index:
                target_pos = i
                break
        
        if target_pos is None:
            return ""
        
        # Get context window
        start_pos = max(0, target_pos - window_size)
        end_pos = min(len(doc_chunks), target_pos + window_size + 1)
        
        context_chunks = doc_chunks[start_pos:end_pos]
        context_text = " ".join([c['chunk_text'] for c in context_chunks])
        
        return context_text
    
    def get_all_documents(self) -> List[str]:
        """
        Get list of all document filenames in the index.
        
        Returns:
            List of unique filenames
        """
        if self.metadata is None:
            return []
        
        return list(set(m['doc_filename'] for m in self.metadata))
    
    def get_document_chunks(self, doc_filename: str) -> List[Dict]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_filename: Document filename
            
        Returns:
            List of chunk metadata
        """
        if self.metadata is None:
            return []
        
        chunks = [
            m for m in self.metadata 
            if m['doc_filename'] == doc_filename
        ]
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x['chunk_index'])
        
        return chunks