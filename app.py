import streamlit as st
from src.query import get_qa_chain
from src.indexer import create_vectorstore

st.set_page_config(page_title="Local Codebase Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Local Codebase Assistant")
st.caption("Fully local • Powered by Ollama • Zero cloud")

with st.sidebar:
    st.header("Index a Codebase")
    
    repo_path = st.text_input(
        "Enter full path of the project folder",
        placeholder=r"D:\programers\work\some-project"
    )
    
    if st.button("Index this folder", type="primary"):
        if not repo_path.strip():
            st.error("Please enter a valid path")
        else:
            with st.spinner("Indexing... this may take 1–3 minutes"):
                try:
                    create_vectorstore(repo_path.strip())
                    st.success("Indexing completed! You can now ask questions.")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    - 100% private
    - Runs fully on your machine
    - Uses small local models
    """)
    st.markdown("**Models:**")
    st.code("qwen2.5-coder:3b\nnomic-embed-text")

@st.cache_resource
def load_chain():
    return get_qa_chain()

try:
    chain = load_chain()
except Exception:
    st.info("No codebase indexed yet. Please index a folder using the sidebar.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the codebase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})