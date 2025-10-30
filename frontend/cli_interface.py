import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DOCUMENTS_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.summarizer import DocumentSummarizer
from src.retriever import DocumentRetriever
from src.qa_system import QASystem
from src.utils import save_json, format_document_info


class CLI:
    """Command-line interface for document summarizer."""
    
    def __init__(self):
        """Initialize CLI."""
        self.doc_processor = DocumentProcessor()
        self.embeddings_manager = None
        self.summarizer = None
        self.retriever = None
        self.qa_system = None
        
        self.documents = []
        self.embeddings_data = None
    
    def print_header(self, text: str):
        """Print formatted header."""
        print("\n" + "=" * 60)
        print(text.center(60))
        print("=" * 60 + "\n")
    
    def print_menu(self):
        """Print main menu."""
        self.print_header("DOCUMENT SUMMARIZER & Q&A SYSTEM")
        print("1. Process Documents")
        print("2. Create/Load Embeddings")
        print("3. Summarize Documents")
        print("4. Ask Questions (Q&A)")
        print("5. View Documents")
        print("6. Exit")
        print()
    
    def process_documents_menu(self):
        """Process documents submenu."""
        self.print_header("PROCESS DOCUMENTS")
        print(f"Documents directory: {DOCUMENTS_DIR}")
        print()
        print("1. Process all documents in directory")
        print("2. Process specific file")
        print("3. Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            print("\nProcessing all documents...")
            self.documents = self.doc_processor.process_directory(DOCUMENTS_DIR)
            print(f"\n✓ Successfully processed {len(self.documents)} documents")
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            filepath = input("\nEnter file path: ").strip()
            try:
                doc_data = self.doc_processor.process_document(Path(filepath))
                self.documents.append(doc_data)
                print(f"\n✓ Successfully processed {doc_data['filename']}")
            except Exception as e:
                print(f"\n✗ Error: {e}")
            input("\nPress Enter to continue...")
    
    def embeddings_menu(self):
        """Embeddings submenu."""
        self.print_header("EMBEDDINGS")
        print("1. Create new embeddings")
        print("2. Load existing embeddings")
        print("3. Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            if not self.documents:
                print("\n✗ No documents loaded. Please process documents first.")
                input("\nPress Enter to continue...")
                return
            
            print("\nInitializing embedding model...")
            if self.embeddings_manager is None:
                self.embeddings_manager = EmbeddingsManager(use_local=True)
            
            print("Creating embeddings...")
            self.embeddings_data = self.embeddings_manager.create_document_embeddings(
                self.documents
            )
            
            # Save embeddings
            embeddings_file = EMBEDDINGS_DIR / "document_embeddings.pkl"
            self.embeddings_manager.save_embeddings(self.embeddings_data, embeddings_file)
            
            # Build index
            self._build_index()
            
            print(f"\n✓ Created and saved embeddings for {self.embeddings_data['num_chunks']} chunks")
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            embeddings_file = EMBEDDINGS_DIR / "document_embeddings.pkl"
            
            if not embeddings_file.exists():
                print(f"\n✗ No embeddings file found at {embeddings_file}")
                input("\nPress Enter to continue...")
                return
            
            print("\nInitializing embedding model...")
            if self.embeddings_manager is None:
                self.embeddings_manager = EmbeddingsManager(use_local=True)
            
            print("Loading embeddings...")
            self.embeddings_data = self.embeddings_manager.load_embeddings(embeddings_file)
            
            if self.embeddings_data:
                # Build index
                self._build_index()
                print(f"\n✓ Loaded {self.embeddings_data['num_chunks']} embeddings")
            else:
                print("\n✗ Failed to load embeddings")
            
            input("\nPress Enter to continue...")
    
    def _build_index(self):
        """Build search index and initialize QA system."""
        print("Building search index...")
        self.retriever = DocumentRetriever(self.embeddings_manager)
        self.retriever.build_index(self.embeddings_data)
        
        print("Initializing Q&A system...")
        self.qa_system = QASystem(self.retriever, use_local=True)
        print("✓ Index built and Q&A system ready")
    
    def summarize_menu(self):
        """Summarization submenu."""
        self.print_header("SUMMARIZE DOCUMENTS")
        
        if not self.documents:
            print("✗ No documents loaded. Please process documents first.")
            input("\nPress Enter to continue...")
            return
        
        print("1. Summarize all documents")
        print("2. Summarize specific document")
        print("3. Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice in ['1', '2']:
            # Initialize summarizer if needed
            if self.summarizer is None:
                print("\nInitializing summarization model...")
                self.summarizer = DocumentSummarizer(use_local=True)
            
            if choice == '1':
                print("\nGenerating summaries for all documents...\n")
                summaries = []
                
                for doc in self.documents:
                    print(f"Summarizing: {doc['filename']}")
                    summary_data = self.summarizer.summarize_document(doc)
                    summaries.append(summary_data)
                    
                    print(f"\n{'-' * 60}")
                    print(f"Document: {doc['filename']}")
                    print(f"{'-' * 60}")
                    print(summary_data['summary'])
                    print(f"{'-' * 60}\n")
                
                # Save summaries
                summaries_file = SUMMARIES_DIR / "all_summaries.json"
                save_json(summaries, summaries_file)
                print(f"✓ Summaries saved to {summaries_file}")
            
            elif choice == '2':
                print("\nAvailable documents:")
                for i, doc in enumerate(self.documents, 1):
                    print(f"{i}. {doc['filename']}")
                
                doc_choice = input("\nSelect document number: ").strip()
                
                try:
                    doc_idx = int(doc_choice) - 1
                    if 0 <= doc_idx < len(self.documents):
                        doc = self.documents[doc_idx]
                        
                        print(f"\nSummarizing: {doc['filename']}")
                        summary_data = self.summarizer.summarize_document(doc)
                        
                        print(f"\n{'-' * 60}")
                        print(f"Document: {doc['filename']}")
                        print(f"{'-' * 60}")
                        print(summary_data['summary'])
                        print(f"{'-' * 60}\n")
                        print(f"Original length: {summary_data['original_length']:,} characters")
                        print(f"Summary length: {summary_data['summary_length']:,} characters")
                        compression = (1 - summary_data['summary_length'] / summary_data['original_length']) * 100
                        print(f"Compression: {compression:.1f}%")
                    else:
                        print("\n✗ Invalid document number")
                except ValueError:
                    print("\n✗ Invalid input")
            
            input("\nPress Enter to continue...")
    
    def qa_menu(self):
        """Q&A interactive menu."""
        if self.qa_system is None:
            print("\n✗ Q&A system not initialized. Please create/load embeddings first.")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("INTERACTIVE Q&A")
        print("Ask questions about your documents.")
        print("Commands:")
        print("  'docs' - Show available documents")
        print("  'back' - Return to main menu")
        print()
        
        available_docs = self.qa_system.get_available_documents()
        
        while True:
            question = input("Question: ").strip()
            
            if question.lower() == 'back':
                break
            
            if question.lower() == 'docs':
                print("\nAvailable documents:")
                for doc in available_docs:
                    print(f"  • {doc}")
                print()
                continue
            
            if not question:
                continue
            
            print("\n" + "⏳ Searching and generating answer...\n")
            
            # Get answer
            response = self.qa_system.answer_question(
                question,
                top_k=3,
                include_sources=True
            )
            
            # Display answer
            print(f"{'-' * 60}")
            print("ANSWER:")
            print(f"{'-' * 60}")
            print(response['answer'])
            print(f"{'-' * 60}")
            print(f"Confidence: {response['confidence']:.2%}")
            
            if response['sources']:
                print("\nSources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['document']} (relevance: {source['relevance_score']:.2%})")
                    print(f"     Snippet: {source['text_snippet'][:150]}...")
            
            print("\n" + "=" * 60 + "\n")
    
    def view_documents_menu(self):
        """View documents menu."""
        self.print_header("DOCUMENT OVERVIEW")
        
        if not self.documents:
            print("No documents loaded.")
            input("\nPress Enter to continue...")
            return
        
        for i, doc in enumerate(self.documents, 1):
            print(f"\n{i}. {doc['filename']}")
            print(f"   Format: {doc['format']}")
            print(f"   Chunks: {doc['num_chunks']}")
            print(f"   Characters: {doc['num_chars']:,}")
            if doc['num_pages']:
                print(f"   Pages: {doc['num_pages']}")
            print(f"   Preview: {doc['full_text'][:150]}...")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the CLI application."""
        while True:
            self.print_menu()
            choice = input("Select option: ").strip()
            
            if choice == '1':
                self.process_documents_menu()
            elif choice == '2':
                self.embeddings_menu()
            elif choice == '3':
                self.summarize_menu()
            elif choice == '4':
                self.qa_menu()
            elif choice == '5':
                self.view_documents_menu()
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print("\n✗ Invalid option. Please try again.")
                input("\nPress Enter to continue...")


def main():
    """Main entry point for CLI."""
    cli = CLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()