from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from .retriever import DocumentRetriever
from config.settings import (
    TINYLLAMA_MODEL_NAME,
    TINYLLAMA_MAX_NEW_TOKENS,
    TINYLLAMA_TEMPERATURE,
    TINYLLAMA_TOP_K,
    TINYLLAMA_TOP_P
)
from pathlib import Path


# Model directory
TINYLLAMA_MODEL_DIR = Path("models/tinyllama")


class QASystem:
    """Question-answering system using TinyLlama chat model."""
    
    def __init__(
        self, 
        retriever: DocumentRetriever,
        use_local: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the QA system with TinyLlama.
        
        Args:
            retriever: DocumentRetriever instance
            use_local: Use locally saved model if available
            device: Device to run model on
        """
        self.retriever = retriever
        self.pipe = None
        self.tokenizer = None
        self.use_local = use_local
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the TinyLlama chat model."""
        try:
            model_path = None
            
            if self.use_local and TINYLLAMA_MODEL_DIR.exists() and list(TINYLLAMA_MODEL_DIR.glob("*")):
                # Load from local directory
                print(f"Loading local TinyLlama model from {TINYLLAMA_MODEL_DIR}")
                model_path = str(TINYLLAMA_MODEL_DIR)
            else:
                # Use HuggingFace model name
                print(f"Loading TinyLlama Q&A model: {TINYLLAMA_MODEL_NAME}")
                model_path = TINYLLAMA_MODEL_NAME
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Pick a safe dtype and device. Use float16 on CUDA (smaller memory) and float32 on CPU.
            # Avoid using `device_map="auto"` by default because it can spawn accelerate and
            # increase memory usage or cause unexpected processes when running inside Streamlit.
            import os

            if torch.cuda.is_available():
                dtype = torch.float16
                device = 0  # first GPU
                device_map = None
            else:
                dtype = torch.float32
                device = -1  # CPU
                device_map = None

            # Allow opt-in to auto device mapping via env var USE_DEVICE_MAP_AUTO=1
            if os.environ.get("USE_DEVICE_MAP_AUTO", "0") == "1":
                device_map = "auto"
                device = None

            # Load model object with low_cpu_mem_usage to reduce memory spikes, then pass into pipeline
            try:
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    low_cpu_mem_usage=True
                )
            except Exception:
                # Fallback to letting pipeline handle loading if direct load fails
                model_obj = None

            if model_obj is not None:
                self.pipe = pipeline(
                    "text-generation",
                    model=model_obj,
                    tokenizer=self.tokenizer,
                    device_map=device_map,
                    device=device
                )
            else:
                # pipeline will load the model (less control over memory)
                self.pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=self.tokenizer,
                    dtype=dtype,
                    device_map=device_map,
                    device=device
                )
            
            print(f"✓ TinyLlama Q&A model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Q&A model: {e}")
            raise
    
    def answer_question(
        self, 
        question: str, 
        top_k: int = 5,
        max_answer_length: int = 300,
        include_sources: bool = True,
        use_context_window: bool = True
    ) -> Dict:
        """
        Answer a question using TinyLlama and retrieved context.
        
        Args:
            question: Question to answer
            top_k: Number of relevant chunks to retrieve
            max_answer_length: Maximum length of answer (for compatibility)
            include_sources: Include source information in response
            use_context_window: Expand chunks to include surrounding context
            Do not provide answers outside retrieved context.
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant chunks
        results = self.retriever.search(question, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Expand context with surrounding chunks if requested
        if use_context_window:
            expanded_results = []
            for result in results:
                try:
                    expanded_context = self.retriever.get_context_window(
                        chunk_index=result['chunk_index'],
                        doc_filename=result['doc_filename'],
                        window_size=1
                    )
                    result['chunk_text'] = expanded_context
                except:
                    pass
                expanded_results.append(result)
            results = expanded_results
        
        # Combine context from top results
        context = self._combine_contexts(results)
        
        # Generate answer using TinyLlama
        answer = self._generate_answer(question, context)
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0.0
        
        # Prepare response
        response = {
            'answer': answer,
            'confidence': float(avg_score)
        }
        
        if include_sources:
            response['sources'] = [
                {
                    'document': r['doc_filename'],
                    'chunk_index': r['chunk_index'],
                    'relevance_score': r['score'],
                    'text_snippet': r['chunk_text'][:200] + "..."
                }
                for r in results
            ]
        
        return response
    
    def _combine_contexts(self, results: List[Dict]) -> str:
        """
        Combine retrieved chunks into a single context.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Combined context text
        """
        contexts = []
        
        for i, result in enumerate(results, 1):
            # Add document name for multi-document awareness
            contexts.append(f"[Document: {result['doc_filename']}]\n{result['chunk_text']}")
        
        return "\n\n".join(contexts)
    
    def _generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Generate answer using TinyLlama chat model.
        
        Args:
            question: Question to answer
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        # Truncate context if too long (TinyLlama has limited context window)
        if len(context) > 1500:
            context = context[:1500] + "..."
        
        # Create chat messages for Q&A
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate. Only use information from the context provided."
            },
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}

Please provide a clear and concise answer based only on the context above.
Do not provide answers outside the provided context.
Do not make up information not contained in the context.
Do not answer questions that cannot be answered with the given context."""
            }
        ]
        
        # Format using chat template
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate answer
        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens=TINYLLAMA_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TINYLLAMA_TEMPERATURE,
                top_k=TINYLLAMA_TOP_K,
                top_p=TINYLLAMA_TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the answer (after the assistant token)
            if "<|assistant|>" in generated_text:
                answer = generated_text.split("<|assistant|>")[-1].strip()
            else:
                # Fallback: get text after the prompt
                answer = generated_text[len(prompt):].strip()
            
            # Clean up
            answer = answer.replace("</s>", "").strip()
            
            # Remove any remaining system tokens
            answer = answer.split("<|")[0].strip()
            
            return answer if answer else "I couldn't generate a clear answer from the context."
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I encountered an error generating the answer."
    
    def ask_with_document(
        self, 
        question: str, 
        doc_filename: str,
        top_k: int = 5,
        max_answer_length: int = 300
    ) -> Dict:
        """
        Answer a question about a specific document.
        
        Args:
            question: Question to answer
            doc_filename: Document to search within
            top_k: Number of relevant chunks to retrieve
            max_answer_length: Maximum length of answer
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant chunks from specific document
        results = self.retriever.search_by_document(question, doc_filename, top_k=top_k)
        
        if not results:
            return {
                'answer': f"I couldn't find relevant information in {doc_filename} to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Combine context
        context = self._combine_contexts(results)
        
        # Generate answer
        answer = self._generate_answer(question, context)
        
        # Calculate confidence
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0.0
        
        return {
            'answer': answer,
            'document': doc_filename,
            'confidence': float(avg_score),
            'sources': [
                {
                    'chunk_index': r['chunk_index'],
                    'relevance_score': r['score'],
                    'text_snippet': r['chunk_text'][:200] + "..."
                }
                for r in results
            ]
        }
    
    def get_available_documents(self) -> List[str]:
        """
        Get list of available documents.
        
        Returns:
            List of document filenames
        """
        return self.retriever.get_all_documents()