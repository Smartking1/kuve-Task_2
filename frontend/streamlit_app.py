import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DOCUMENTS_DIR
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.summarizer import DocumentSummarizer
from src.retriever import DocumentRetriever
from src.qa_system import QASystem


# Page configuration
st.set_page_config(
    page_title="Multi-Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_embeddings_manager():
    """Load only the embeddings manager (cached).

    We intentionally do NOT load the TinyLlama summarizer at app start because
    it's large and can cause Streamlit to OOM or spawn background processes.
    The summarizer is created on-demand when processing documents.
    """
    with st.spinner("🤖 Loading embedding model... This may take a moment on first run."):
        embeddings_manager = EmbeddingsManager(use_local=True)
    return embeddings_manager


@st.cache_resource
def get_summarizer():
    """Create (and cache) the DocumentSummarizer when needed."""
    return DocumentSummarizer(use_local=True)


def process_csv_file(file):
    """Process CSV file and convert to text format."""
    try:
        df = pd.read_csv(file)
        
        # Convert DataFrame to readable text
        text_parts = []
        
        # Add header info
        text_parts.append(f"CSV File: {file.name}")
        text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        text_parts.append(f"Number of rows: {len(df)}")
        text_parts.append("\n--- Data ---\n")
        
        # Convert to text representation
        # Include column headers and first N rows
        text_parts.append(df.to_string(max_rows=100))
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("\n--- Summary Statistics ---\n")
            text_parts.append(df[numeric_cols].describe().to_string())
        
        return '\n'.join(text_parts)
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")


def process_uploaded_file(file, doc_processor):
    """Process an uploaded file and return document data."""
    file_suffix = Path(file.name).suffix.lower()
    
    # Handle CSV files specially
    if file_suffix == '.csv':
        try:
            text = process_csv_file(file)
            
            # Create document data structure
            doc_data = {
                'filename': file.name,
                'original_filename': file.name,
                'filepath': file.name,
                'format': '.csv',
                'full_text': text,
                'chunks': [],
                'num_chunks': 0,
                'num_chars': len(text),
                'num_pages': None
            }
            
            # Chunk the text
            from src.utils import chunk_text
            chunks = chunk_text(text, chunk_size=512, overlap=50)
            doc_data['chunks'] = chunks
            doc_data['num_chunks'] = len(chunks)
            
            return doc_data
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    # Handle other file types (PDF, TXT, DOCX)
    else:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process document
            doc_data = doc_processor.process_document(tmp_path)
            doc_data['original_filename'] = file.name
            return doc_data
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)


def main():
    """Main Streamlit app."""
    
    # Custom CSS
    st.markdown("""
        <style>
        .summary-box {
            background-color: #000000;
            color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .qa-box {
            background-color: #000000;
            color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .doc-card {
            background-color: #000000;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
        }
        .success-box {
            background-color: #000000;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📚 Multi-Document Q&A System")
    st.markdown("""
    Upload multiple documents, get instant summaries, and ask questions across all your documents!
    
    **Supported formats:** PDF, TXT, DOCX, CSV
    """)
    
    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'embeddings_data' not in st.session_state:
        st.session_state.embeddings_data = None
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    # Load embeddings manager only (summarizer is created on-demand)
    embeddings_manager = load_embeddings_manager()
    doc_processor = DocumentProcessor()
    
    # Sidebar: Document Upload and Management
    with st.sidebar:
        st.markdown("## 📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple PDF, TXT, DOCX, or CSV files"
        )
        
        if uploaded_files:
            # Show upload count
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files:
                st.info(f"📊 {len(new_files)} new file(s) ready to process")
                
                # Process button
                if st.button("🔄 Process All Documents", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_files = len(new_files)
                    
                    for idx, file in enumerate(new_files):
                        status_text.text(f"Processing {idx+1}/{total_files}: {file.name}")
                        
                        try:
                            # Process document
                            doc_data = process_uploaded_file(file, doc_processor)
                            
                            # Add to documents list
                            st.session_state.documents.append(doc_data)
                            st.session_state.processed_files.add(file.name)
                            
                            # Defer summary generation to a separate step to avoid loading the
                            # heavy TinyLlama model during the initial processing step.
                            status_text.text(f"Processed (summary pending): {file.name}")
                            st.session_state.summaries[file.name] = {
                                'filename': file.name,
                                'summary': "(Summary pending) Click 'Generate Summaries' to load the model and generate.",
                                'original_length': len(doc_data['full_text']),
                                'summary_length': 0
                            }
                            
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / total_files)
                    
                    # Create embeddings for all documents
                    if st.session_state.documents:
                        status_text.text("Creating embeddings for Q&A...")
                        embeddings_data = embeddings_manager.create_document_embeddings(
                            st.session_state.documents
                        )
                        st.session_state.embeddings_data = embeddings_data

                        # Build retriever now but DO NOT instantiate the heavy QASystem yet.
                        # We'll create the QASystem on-demand when the user asks the first question.
                        retriever = DocumentRetriever(embeddings_manager)
                        retriever.build_index(embeddings_data)
                        st.session_state.retriever = retriever
                    
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Processed {total_files} document(s)!")
                    st.rerun()
        
        # Show processed documents
        if st.session_state.documents:
            st.markdown("---")
            st.markdown(f"## 📚 Loaded Documents ({len(st.session_state.documents)})")
            
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['original_filename']}"):
                    st.write(f"**Format:** {doc['format']}")
                    st.write(f"**Chunks:** {doc['num_chunks']}")
                    st.write(f"**Size:** {doc['num_chars']:,} characters")
                    if doc['num_pages']:
                        st.write(f"**Pages:** {doc['num_pages']}")

            # Button to generate summaries on-demand (loads TinyLlama)
            st.markdown("---")
            st.markdown(
                "**Summaries are deferred** — generating summaries will load the TinyLlama model and may use significant memory."
            )
            if st.button("🧾 Generate Summaries (load model)", type="primary", use_container_width=True):
                # Create summarizer and generate summaries with progress
                try:
                    with st.spinner("⚙️ Loading summarization model and generating summaries..."):
                        summarizer = get_summarizer()
                        total = len(st.session_state.documents)
                        p = st.progress(0)
                        for idx, doc in enumerate(st.session_state.documents):
                            st.session_state.summaries[doc['original_filename']] = summarizer.summarize_document(doc)
                            p.progress((idx + 1) / total)
                    st.success("✅ Summaries generated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate summaries: {e}")
        
        # Reset button
        if st.session_state.documents:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                st.session_state.documents = []
                st.session_state.summaries = {}
                st.session_state.embeddings_data = None
                st.session_state.qa_system = None
                st.session_state.retriever = None
                st.session_state.chat_history = []
                st.session_state.processed_files = set()
                # Optionally keep cached summarizer to avoid reloading; if you want to free memory,
                # remove it explicitly:
                # st.session_state.pop('summarizer', None)
                st.rerun()
    
    # Main content area
    if not st.session_state.documents:
        # Welcome screen
        st.markdown("""
        <div class="success-box">
        <h3>👋 Welcome! Get Started in 3 Easy Steps:</h3>
        <ol>
            <li><strong>Upload</strong> your documents using the sidebar (PDF, TXT, DOCX, CSV)</li>
            <li><strong>Process</strong> them to generate summaries and enable Q&A</li>
            <li><strong>Ask questions</strong> and get AI-powered answers from your documents</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI-Powered
            - Automatic summarization
            - Semantic understanding
            - Context-aware answers
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Multi-Format
            - PDF documents
            - Text files
            - Word documents
            - CSV data files
            """)
        
        with col3:
            st.markdown("""
            ### 🔒 Privacy First
            - 100% offline processing
            - No data sent to cloud
            - Secure and private
            """)
    
    else:
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📝 Summaries", "💬 Ask Questions", "📊 Document Details"])
        
        # Tab 1: Summaries
        with tab1:
            st.markdown("## 📝 Document Summaries")
            st.markdown("AI-generated summaries of your uploaded documents:")
            
            for doc in st.session_state.documents:
                filename = doc['original_filename']
                
                if filename in st.session_state.summaries:
                    summary_data = st.session_state.summaries[filename]
                    
                    with st.container():
                        st.markdown(f"### 📄 {filename}")
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Length", f"{summary_data['original_length']:,} chars")
                        with col2:
                            st.metric("Summary Length", f"{summary_data['summary_length']:,} chars")
                        with col3:
                            compression = (1 - summary_data['summary_length'] / summary_data['original_length']) * 100
                            st.metric("Compression", f"{compression:.1f}%")
                        
                        # Summary text
                        st.markdown(f"""
                        <div class="summary-box">
                        {summary_data['summary']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
        
        # Tab 2: Q&A Interface
        with tab2:
            st.markdown("## 💬 Ask Questions About Your Documents")
            
            # Q&A UI: create QASystem on-demand when the user asks a question
            if 'retriever' not in st.session_state:
                st.warning("⚠️ Please process documents first to enable Q&A functionality.")
            else:
                # Quick question suggestions
                st.markdown("**💡 Try asking:**")
                suggestions = [
                    "What are the main points across all documents?",
                    "Summarize the key findings",
                    "What data is contained in the CSV files?",
                    "Compare information across documents"
                ]
                
                cols = st.columns(4)
                for i, suggestion in enumerate(suggestions):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                            st.session_state.current_question = suggestion
                
                st.markdown("---")
                
                # Question input
                question = st.text_input(
                    "🔍 Your question:",
                    placeholder="Ask anything about your documents...",
                    key="question_input"
                )
                
                # Use suggestion if clicked
                if 'current_question' in st.session_state:
                    question = st.session_state.current_question
                    del st.session_state.current_question
                
                # Ask button
                col1, col2 = st.columns([1, 5])
                with col1:
                    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
                with col2:
                    if st.button("🗑️ Clear History", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
                
                if ask_button and question:
                    with st.spinner("🤔 Finding answer..."):
                        try:
                            # Lazily create the QASystem so the model is only loaded when necessary
                            if 'qa_system' not in st.session_state or st.session_state.get('qa_system') is None:
                                try:
                                    st.info("Loading Q&A model (this may take a minute)...")
                                    st.session_state.qa_system = QASystem(st.session_state.retriever, use_local=True)
                                except Exception as e:
                                    st.error(f"❌ Failed to load Q&A model: {e}")
                                    raise

                            response = st.session_state.qa_system.answer_question(
                                question,
                                top_k=3,
                                include_sources=True
                            )

                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': question,
                                'answer': response['answer'],
                                'confidence': response['confidence'],
                                'sources': response.get('sources', [])
                            })

                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error generating answer: {str(e)}")
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("---")
                    st.markdown("### 💭 Conversation History")
                    
                    # Show newest first
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.container():
                            st.markdown(f"""
                            <div class="qa-box">
                            <strong>❓ Q:</strong> {chat['question']}<br><br>
                            <strong>💡 A:</strong> {chat['answer']}<br><br>
                            <small>📊 Confidence: {chat['confidence']:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show sources in expander
                            if chat['sources']:
                                with st.expander(f"📚 View {len(chat['sources'])} Source(s)"):
                                    for j, source in enumerate(chat['sources'], 1):
                                        st.markdown(f"**Source {j}: {source['document']}**")
                                        st.markdown(f"*Relevance: {source['relevance_score']:.1%}*")
                                        st.text(source['text_snippet'])
                                        if j < len(chat['sources']):
                                            st.markdown("---")
                            
                            st.markdown("")  # Spacing
        
        # Tab 3: Document Details
        with tab3:
            st.markdown("## 📊 Document Details")
            
            # Overall statistics
            total_chunks = sum(doc['num_chunks'] for doc in st.session_state.documents)
            total_chars = sum(doc['num_chars'] for doc in st.session_state.documents)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Total Documents", len(st.session_state.documents))
            with col2:
                st.metric("🧩 Total Chunks", f"{total_chunks:,}")
            with col3:
                st.metric("📏 Total Characters", f"{total_chars:,}")
            
            st.markdown("---")
            
            # Individual document details
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['original_filename']}", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Format:** {doc['format']}")
                        st.write(f"**Chunks:** {doc['num_chunks']}")
                        st.write(f"**Characters:** {doc['num_chars']:,}")
                        if doc['num_pages']:
                            st.write(f"**Pages:** {doc['num_pages']}")
                    
                    with col2:
                        st.text_area(
                            "Preview (first 500 characters):",
                            doc['full_text'][:500] + "...",
                            height=150,
                            key=f"preview_{doc['original_filename']}"
                        )


if __name__ == "__main__":
    main()