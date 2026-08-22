# Local Codebase Assistant

A fully local AI coding assistant that indexes any codebase and answers questions about it using small open-source models.

**100% private • Zero cloud • Runs on modest hardware (tested on 16GB RAM + 4GB VRAM)**

## Features

- Index any local project folder
- AST-aware chunking for Python (functions & classes as separate units)
- Specialized modes:
  - General Question
  - Explain Function/Class
  - Architecture Overview
  - Suggest Improvements
- Optional filtering (Functions only / Classes only)
- Clean Streamlit interface
- Fully offline

## Tech Stack

- **LLM**: `qwen2.5-coder:3b`
- **Embeddings**: `nomic-embed-text`
- **Vector Store**: Chroma
- **Framework**: LangChain + Streamlit
- **Code parsing**: Python AST

## How to Run

1. Make sure Ollama is running and models are pulled:
   ```bash
   ollama pull qwen2.5-coder:3b
   ollama pull nomic-embed-text

2. Activate environment and start:Bashconda activate codeassist
   streamlit run app.py
   
   In the sidebar, paste the full path of any project and click Index this folder.

3.Choose a mode + optional filter, then ask questions.

Project Structure
codebase-assistant/
├── app.py                 # Streamlit UI
├── src/
│   ├── indexer.py         # AST-aware indexing
│   └── query.py           # Retrieval + filtering
├── data/                  # Vector stores (auto-generated)
└── README.md

> Fully local AI coding assistant with AST-aware indexing, specialized query modes, and metadata filtering.

## Project Structure

codebase-assistant/
├── app.py                  # Streamlit UI with modes & filters
├── src/
│   ├── indexer.py          # AST-aware code indexing
│   └── query.py            # Retrieval + metadata filtering
├── data/                   # Auto-generated vector stores
├── requirements.txt
├── LICENSE
└── README.md


## Notes

- Designed to run fully offline on modest hardware (tested on 16GB RAM + 4GB VRAM).
- Uses small open-source models only.
- No cloud APIs or paid services required.