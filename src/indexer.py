import os
import time
import ast
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CODE_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".md", ".txt", ".json", ".yaml", ".yml"]
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", "data", ".idea", ".vscode"}

def extract_python_chunks(file_path: str, content: str):
    """Extract functions and classes as separate high-quality chunks."""
    chunks = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                code = "\n".join(content.splitlines()[start:end])
                
                name = node.name
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                
                metadata = {
                    "source": file_path,
                    "name": name,
                    "type": kind,
                    "start_line": node.lineno
                }
                chunks.append(Document(page_content=code, metadata=metadata))
    except Exception:
        # If AST fails, fall back to normal splitting later
        pass
    return chunks

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
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                if ext == ".py":
                    # Use AST for Python
                    py_chunks = extract_python_chunks(str(file_path), content)
                    if py_chunks:
                        documents.extend(py_chunks)
                        continue
                
                # For non-Python or failed AST → normal document
                documents.append(Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                ))
            except Exception:
                continue
    return documents

def create_vectorstore(repo_path: str):
    timestamp = int(time.time())
    persist_directory = f"data/chroma_db_{timestamp}"
    
    print(f"Loading files from: {repo_path}")
    documents = load_codebase(repo_path)
    
    if not documents:
        raise ValueError("No supported files found!")

    print(f"Found {len(documents)} code units. Creating embeddings...")
    
    # Further split very large chunks if needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    with open("data/latest_db.txt", "w") as f:
        f.write(persist_directory)
    
    print(f"Vector store saved to: {persist_directory}")
    return persist_directory