
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config.settings import (
    TINYLLAMA_MODEL_NAME,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH,
    TINYLLAMA_MAX_NEW_TOKENS,
    TINYLLAMA_TEMPERATURE,
    TINYLLAMA_TOP_K,
    TINYLLAMA_TOP_P
)


# Model directory
TINYLLAMA_MODEL_DIR = Path("models/tinyllama")
TINYLLAMA_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class DocumentSummarizer:
    """Summarize documents using TinyLlama."""
    
    def __init__(self, use_local: bool = True, device: Optional[str] = None):
        """
        Initialize the summarizer.
        
        Args:
            use_local: Use locally saved model if available
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.pipe = None
        self.tokenizer = None
        self.model = None
        self.use_local = use_local
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the TinyLlama model."""
        try:
            model_path = None
            
            if self.use_local and TINYLLAMA_MODEL_DIR.exists() and list(TINYLLAMA_MODEL_DIR.glob("*")):
                # Load from local directory
                print(f"Loading local TinyLlama model from {TINYLLAMA_MODEL_DIR}")
                model_path = str(TINYLLAMA_MODEL_DIR)
            else:
                # Use HuggingFace model name
                print(f"Loading TinyLlama model: {TINYLLAMA_MODEL_NAME}")
                print("TinyLlama is ~2.2GB - first download may take a few minutes...")
                model_path = TINYLLAMA_MODEL_NAME
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Pick a safe dtype and device. Use float16 on CUDA (smaller memory) and float32 on CPU.
            # Avoid using `device_map="auto"` by default because it can spawn accelerate and
            # increase memory usage or cause unexpected processes when running inside Streamlit.
            import os

            if torch.cuda.is_available():
                dtype = torch.float16
                device = 0
                device_map = None
            else:
                dtype = torch.float32
                device = -1
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
            
            print(f"✓ TinyLlama model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading TinyLlama model: {e}")
            raise
    
    def summarize_text(
        self, 
        text: str, 
        max_length: int = MAX_SUMMARY_LENGTH,
        min_length: int = MIN_SUMMARY_LENGTH,
        max_new_tokens: int = TINYLLAMA_MAX_NEW_TOKENS,
        temperature: float = TINYLLAMA_TEMPERATURE
    ) -> str:
        """
        Summarize a single text using TinyLlama.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (for compatibility)
            min_length: Minimum length of summary (for compatibility)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Summary text
        """
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        # Truncate text if too long (TinyLlama has limited context)
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        # Create chat messages for summarization
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise, accurate summaries of documents."
            },
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
            }
        ]
        
        # Format using chat template
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate summary
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=TINYLLAMA_TOP_K,
            top_p=TINYLLAMA_TOP_P,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        # Extract just the summary (after the assistant token)
        if "<|assistant|>" in generated_text:
            summary = generated_text.split("<|assistant|>")[-1].strip()
        else:
            # Fallback: get text after the prompt
            summary = generated_text[len(prompt):].strip()
        
        # Clean up
        summary = summary.replace("</s>", "").strip()
        
        return summary
    
    def summarize_document(
        self, 
        doc_data: Dict,
        max_length: int = MAX_SUMMARY_LENGTH,
        min_length: int = MIN_SUMMARY_LENGTH
    ) -> Dict:
        """
        Summarize an entire document.
        
        Args:
            doc_data: Document data dictionary
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        print(f"Summarizing document: {doc_data['filename']}")
        
        text = doc_data['full_text']
        
        # For long documents, use hierarchical summarization
        if len(text) > 3000:
            summary = self._hierarchical_summarization(
                doc_data['chunks'], 
                max_length, 
                min_length
            )
        else:
            summary = self.summarize_text(text, max_length, min_length)
        
        return {
            'filename': doc_data['filename'],
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
        }
    
    def _hierarchical_summarization(
        self, 
        chunks: List[str], 
        max_length: int,
        min_length: int
    ) -> str:
        """
        Perform hierarchical summarization for long documents.
        
        Args:
            chunks: List of text chunks
            max_length: Maximum length of final summary
            min_length: Minimum length of final summary
            
        Returns:
            Final summary
        """
        # Summarize each chunk
        chunk_summaries = []
        
        # Only summarize first 5 chunks to avoid too much processing
        for i, chunk in enumerate(chunks[:5]):
            if len(chunk) > 200:  # Only summarize substantial chunks
                try:
                    summary = self.summarize_text(
                        chunk, 
                        max_length=150, 
                        min_length=30,
                        max_new_tokens=150
                    )
                    chunk_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing chunk {i}: {e}")
                    continue
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # Final summarization
        if len(combined_summary) > 1000:
            final_summary = self.summarize_text(
                combined_summary,
                max_length=max_length,
                min_length=min_length
            )
        else:
            final_summary = combined_summary
        
        return final_summary