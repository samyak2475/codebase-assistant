from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def get_qa_chain(filter_type: str = None):
    """
    filter_type can be: None, "function", or "class"
    """
    latest_file = "data/latest_db.txt"
    
    if not os.path.exists(latest_file):
        raise FileNotFoundError("No codebase has been indexed yet.")
    
    with open(latest_file, "r") as f:
        persist_directory = f.read().strip()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Apply metadata filter if requested
    search_kwargs = {"k": 5}
    if filter_type in ["function", "class"]:
        search_kwargs["filter"] = {"type": filter_type}
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    llm = ChatOllama(
        model="qwen2.5-coder:3b",
        temperature=0.1
    )
    
    template = """You are a helpful local coding assistant running fully offline.

Use the provided code context to answer the question.
Be specific and refer to actual function/class names when possible.

Context from the codebase:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain