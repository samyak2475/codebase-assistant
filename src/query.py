from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(persist_directory: str = "data/chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOllama(
        model="qwen2.5-coder:3b",   # change to "llama3.2:3b" if this model is missing
        temperature=0.1
    )
    
    template = """You are a helpful coding assistant. 
Answer the question using only the provided code context.
If you don't know, say you don't know. Be concise and clear.

Context:
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

if __name__ == "__main__":
    chain = get_qa_chain()
    
    print("\n=== Local Codebase Assistant ===")
    print("Type your question (or 'quit' to exit)\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
            
        print("\nThinking...")
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")