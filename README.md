# Local Codebase Assistant

A fully local AI coding assistant that indexes any codebase and answers questions about it using small open-source models.

**100% private • Zero cloud • Runs on modest hardware**

## Features

- Index any local project folder
- Ask natural language questions about the code
- Powered by Ollama + Chroma
- Clean Streamlit interface
- Works offline

## Tech Stack

- **LLM**: `qwen2.5-coder:3b`
- **Embeddings**: `nomic-embed-text`
- **Vector Store**: Chroma
- **Framework**: LangChain + Streamlit
- **Hardware tested**: ASUS A15 (Ryzen 7 + 16GB RAM + 4GB VRAM)

## How to Run

1. Make sure Ollama is running and the models are pulled:
   ```bash
   ollama pull qwen2.5-coder:3b
   ollama pull nomic-embed-text
   
2. Activate the environment and start the app:Bashconda activate codeassist
    streamlit run app.py

3. In the sidebar, paste the full path of any project folder and click Index this folder.

4. Start asking questions.