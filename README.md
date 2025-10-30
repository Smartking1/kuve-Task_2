# Kuve Multi-Document Q&A System

## Overview

Kuve is an offline, privacy-first multi-document Q&A and summarization system. Upload documents (PDF, TXT, DOCX, CSV), generate instant summaries, and ask questions across all your files using TinyLlama and semantic search.

## Features

- **AI-powered Q&A**: Ask questions and get context-aware answers from your documents.
- **Automatic Summarization**: Generate concise summaries for each document.
- **Multi-format Support**: PDF, TXT, DOCX, CSV.
- **Offline Processing**: No data sent to the cloud; all models run locally.
- **Fast Embeddings**: Uses MiniLM for semantic search.
- **Streamlit UI**: Modern, interactive web interface.

## Quick Start

### 1. Install Requirements

```powershell
pip install -r requirements.txt
```

### 2. Download Models (Optional for offline use)

```powershell
python models/download_models.py
```

### 3. Run the App

```powershell
streamlit run frontend/streamlit_app.py
```

### 4. Usage

- Upload documents in the sidebar.
- Click **Process All Documents** to extract text and create embeddings.
- Click **Generate Summaries** to load TinyLlama and summarize documents (may take time and memory).
- Ask questions in the Q&A tab.

## Troubleshooting

- **Streamlit disconnects or crashes when loading TinyLlama**:
	- Model loading is deferred until you click **Generate Summaries** or ask a question.
	- If you run out of memory, try CPU-only mode:
		```powershell
		$env:CUDA_VISIBLE_DEVICES = ""
		streamlit run frontend/streamlit_app.py
		```
	- For large documents, process and summarize one at a time.
- **pip not found in virtual environment**:
	- Reinstall pip using get-pip.py:
		```powershell
		curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
		.\kuve\Scripts\python.exe get-pip.py
		Remove-Item get-pip.py
		```
- **Accelerate not installed**:
	- Install with:
		```powershell
		pip install accelerate
		```

## Project Structure

```
main.py
requirements.txt
config/
	settings.py
frontend/
	cli_interface.py
	streamlit_app.py
kuve/
	(virtual environment)
models/
	download_models.py
src/
	document_processor.py
	embeddings_manager.py
	qa_system.py
	retriever.py
	summarizer.py
	utils.py
```

## .gitignore

See `.gitignore` for recommended patterns. Models, virtual environments, and cache files are excluded by default.


