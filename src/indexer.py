import os
import time
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CODE_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".md", ".txt", ".json", ".yaml", ".yml"]
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", "data", ".idea", ".vscode"}

def load_codebase(repo_path: str):
    documents = []
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        raise ValueError(f"Path does not exist: {repo_path}")
    
    for ext in CODE_EXTENSIONS:
        for file_path in repo_path.rglob(f"*{ext}"):
            if any(ignored in file_path.parts for ignored in IGNORE_DIRS):
                continue
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception:
                continue
    return documents

def create_vectorstore(repo_path: str):
    """Creates a brand new vector store every time (avoids Windows lock)."""
    
    # Create unique folder name using timestamp
    timestamp = int(time.time())
    persist_directory = f"data/chroma_db_{timestamp}"
    
    print(f"Loading files from: {repo_path}")
    documents = load_codebase(repo_path)
    
    if not documents:
        raise ValueError("No supported files found in the given path!")

    print(f"Found {len(documents)} files. Splitting into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    print("Creating embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Save the latest path so query can find it
    with open("data/latest_db.txt", "w") as f:
        f.write(persist_directory)
    
    print(f"Vector store saved to: {persist_directory}")
    return persist_directory