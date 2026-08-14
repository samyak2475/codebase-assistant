import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Supported code file extensions
CODE_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".md", ".txt", ".json", ".yaml", ".yml"]

def load_codebase(repo_path: str):
    """Load all supported code files from a folder."""
    documents = []
    for ext in CODE_EXTENSIONS:
        loader = DirectoryLoader(
            repo_path,
            glob=f"**/*{ext}",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        docs = loader.load()
        documents.extend(docs)
    return documents

def create_vectorstore(repo_path: str, persist_directory: str = "data/chroma_db"):
    """Index the codebase and save the vector store."""
    print(f"Loading files from: {repo_path}")
    documents = load_codebase(repo_path)
    
    if not documents:
        print("No supported files found!")
        return None

    print(f"Found {len(documents)} files. Splitting into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    print("Creating embeddings (this may take a minute)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store saved to: {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    # Test with a small local folder (change this path later)
    test_path = r"D:\programers\work\codebase-assistant"   # current project for testing
    create_vectorstore(test_path)