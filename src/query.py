from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def get_qa_chain():
    # Read the latest database path
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
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOllama(
        model="qwen2.5-coder:3b",
        temperature=0.1
    )
    
    template = """You are a helpful local coding assistant running fully offline.

Use the provided code context to answer questions about the project whenever possible.
If the question is general or the context does not contain the answer, you can still reply helpfully but clearly state when you are not using the codebase.

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