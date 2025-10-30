import os
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
from .utils import clean_text, chunk_text


class DocumentProcessor:
    """Process and extract text from documents."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = ['.pdf', '.txt', '.docx', '.csv']
    
    def is_supported(self, filepath: Path) -> bool:
        """
        Check if file format is supported.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if supported, False otherwise
        """
        return filepath.suffix.lower() in self.supported_formats
    
    def extract_text_from_pdf(self, filepath: Path, use_pdfplumber: bool = True) -> str:
        """
        Extract text from PDF file.
        
        Args:
            filepath: Path to PDF file
            use_pdfplumber: Use pdfplumber (better) or PyPDF2
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            if use_pdfplumber:
                # pdfplumber - better for complex layouts
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                # PyPDF2 - faster but less accurate
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF {filepath}: {e}")
            return ""
        
        return text
    
    def extract_text_from_txt(self, filepath: Path) -> str:
        """
        Extract text from TXT file.
        
        Args:
            filepath: Path to TXT file
            
        Returns:
            Extracted text
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading TXT file {filepath}: {e}")
                return ""
        except Exception as e:
            print(f"Error reading TXT file {filepath}: {e}")
            return ""
    
    def extract_text_from_docx(self, filepath: Path) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            filepath: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            doc = DocxDocument(filepath)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {filepath}: {e}")
            return ""
    
    def process_document(
        self, 
        filepath: Path, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> Dict:
        """
        Process a document and extract text chunks.
        
        Args:
            filepath: Path to document
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            Dictionary containing document data
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not self.is_supported(filepath):
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Extract text based on file type
        suffix = filepath.suffix.lower()
        if suffix == '.pdf':
            text = self.extract_text_from_pdf(filepath)
        elif suffix == '.txt':
            text = self.extract_text_from_txt(filepath)
        elif suffix == '.docx':
            text = self.extract_text_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Clean text
        text = clean_text(text)
        
        if not text:
            raise ValueError(f"No text extracted from {filepath}")
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        # Get document metadata
        num_pages = self._get_num_pages(filepath)
        
        return {
            'filename': filepath.name,
            'filepath': str(filepath),
            'format': suffix,
            'full_text': text,
            'chunks': chunks,
            'num_chunks': len(chunks),
            'num_chars': len(text),
            'num_pages': num_pages
        }
    
    def _get_num_pages(self, filepath: Path) -> Optional[int]:
        """
        Get number of pages in document.
        
        Args:
            filepath: Path to document
            
        Returns:
            Number of pages or None
        """
        suffix = filepath.suffix.lower()
        
        if suffix == '.pdf':
            try:
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages)
            except:
                return None
        
        return None
    
    def process_directory(
        self, 
        directory: Path, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> List[Dict]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory: Path to directory
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List of document data dictionaries
        """
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        documents = []
        
        for filepath in directory.iterdir():
            if filepath.is_file() and self.is_supported(filepath):
                try:
                    doc_data = self.process_document(filepath, chunk_size, overlap)
                    documents.append(doc_data)
                    print(f"✓ Processed: {filepath.name}")
                except Exception as e:
                    print(f"✗ Error processing {filepath.name}: {e}")
        
        return documents