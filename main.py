import argparse
from pathlib import Path
from config.settings import DOCUMENTS_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.summarizer import DocumentSummarizer
from src.retriever import DocumentRetriever
from src.qa_system import QASystem
from src.utils import save_json


class DocumentSummarizerApp:
    """Main application for document summarization and Q&A."""
    
    def __init__(self):
        """Initialize the application."""
        print("=" * 60)
        print("Intelligent Document Summarizer with Contextual Q&A")
        print("=" * 60)
        print()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.embeddings_manager = EmbeddingsManager(use_local=True)
        self.summarizer = DocumentSummarizer(use_local=True)
        self.retriever = None
        self.qa_system = None
        
        self.documents = []
        self.embeddings_data = None
    
    def process_documents(self, directory: Path = DOCUMENTS_DIR):
        """
        Process all documents in the directory.
        
        Args:
            directory: Path to documents directory
        """
        print(f"Processing documents from: {directory}")
        print("-" * 60)
        
        self.documents = self.doc_processor.process_directory(directory)
        
        print(f"\n✓ Processed {len(self.documents)} documents")
        print()
    
    def create_embeddings(self):
        """Create embeddings for all documents."""
        if not self.documents:
            print("No documents to process. Please run process_documents first.")
            return
        
        print("Creating embeddings...")
        print("-" * 60)
        
        self.embeddings_data = self.embeddings_manager.create_document_embeddings(
            self.documents
        )
        
        # Save embeddings
        embeddings_file = EMBEDDINGS_DIR / "document_embeddings.pkl"
        self.embeddings_manager.save_embeddings(self.embeddings_data, embeddings_file)
        
        print(f"✓ Created embeddings for {self.embeddings_data['num_chunks']} chunks")
        print()
    
    def load_embeddings(self, filepath: Path = None):
        """
        Load existing embeddings.
        
        Args:
            filepath: Path to embeddings file
        """
        if filepath is None:
            filepath = EMBEDDINGS_DIR / "document_embeddings.pkl"
        
        self.embeddings_data = self.embeddings_manager.load_embeddings(filepath)
        
        if self.embeddings_data:
            print(f"✓ Loaded {self.embeddings_data['num_chunks']} embeddings")
    
    def build_search_index(self):
        """Build search index for retrieval."""
        if self.embeddings_data is None:
            print("No embeddings available. Please create or load embeddings first.")
            return
        
        print("Building search index...")
        print("-" * 60)
        
        self.retriever = DocumentRetriever(self.embeddings_manager)
        self.retriever.build_index(self.embeddings_data)
        
        # Initialize QA system
        self.qa_system = QASystem(self.retriever, use_local=True)
        
        print("✓ Search index ready")
        print()
    
    def summarize_all_documents(self):
        """Summarize all processed documents."""
        if not self.documents:
            print("No documents to summarize.")
            return
        
        print("Generating summaries...")
        print("-" * 60)
        
        summaries = []
        
        for doc in self.documents:
            summary_data = self.summarizer.summarize_document(doc)
            summaries.append(summary_data)
            
            print(f"\n{doc['filename']}:")
            print(f"{summary_data['summary']}")
            print()
        
        # Save summaries
        summaries_file = SUMMARIES_DIR / "all_summaries.json"
        save_json(summaries, summaries_file)
        
        print(f"✓ Summaries saved to {summaries_file}")
        print()
    
    def interactive_qa(self):
        """Start interactive Q&A session."""
        if self.qa_system is None:
            print("Q&A system not initialized. Please build search index first.")
            return
        
        print("=" * 60)
        print("Interactive Q&A Mode")
        print("=" * 60)
        print("Ask questions about your documents.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'docs' to see available documents.")
        print()
        
        available_docs = self.qa_system.get_available_documents()
        
        while True:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'docs':
                print("\nAvailable documents:")
                for doc in available_docs:
                    print(f"  - {doc}")
                print()
                continue
            
            if not question:
                continue
            
            print("\nSearching and generating answer...\n")
            
            # Get answer
            response = self.qa_system.answer_question(
                question,
                top_k=3,
                include_sources=True
            )
            
            # Display answer
            print(f"Answer: {response['answer']}")
            print(f"\nConfidence: {response['confidence']:.2%}")
            
            if response['sources']:
                print("\nSources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['document']} (relevance: {source['relevance_score']:.2%})")
            
            print("\n" + "-" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Summarizer with Contextual Q&A"
    )
    parser.add_argument(
        '--mode',
        choices=['process', 'summarize', 'qa', 'full'],
        default='full',
        help='Operation mode'
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=DOCUMENTS_DIR,
        help='Directory containing documents'
    )
    parser.add_argument(
        '--load-embeddings',
        action='store_true',
        help='Load existing embeddings instead of creating new ones'
    )
    
    args = parser.parse_args()
    
    app = DocumentSummarizerApp()
    
    if args.mode == 'process':
        # Process documents only
        app.process_documents(args.docs_dir)
        app.create_embeddings()
    
    elif args.mode == 'summarize':
        # Summarize documents
        app.process_documents(args.docs_dir)
        app.summarize_all_documents()
    
    elif args.mode == 'qa':
        # Q&A mode
        if args.load_embeddings:
            app.load_embeddings()
        else:
            app.process_documents(args.docs_dir)
            app.create_embeddings()
        
        app.build_search_index()
        app.interactive_qa()
    
    elif args.mode == 'full':
        # Full pipeline
        if args.load_embeddings:
            app.load_embeddings()
        else:
            app.process_documents(args.docs_dir)
            app.create_embeddings()
        
        app.build_search_index()
        
        # Summarize
        if app.documents:
            app.summarize_all_documents()
        
        # Interactive Q&A
        app.interactive_qa()


if __name__ == "__main__":
    main()