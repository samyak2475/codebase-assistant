import streamlit as st
from src.query import get_qa_chain

st.set_page_config(page_title="Local Codebase Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Local Codebase Assistant")
st.caption("Fully local • Powered by Ollama • Zero cloud")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool indexes any local codebase and lets you ask questions about it.
    
    - 100% private
    - Runs on your machine
    - Uses small local models
    """)
    st.markdown("---")
    st.markdown("**Models used:**")
    st.code("qwen2.5-coder:3b\nnomic-embed-text")

@st.cache_resource
def load_chain():
    return get_qa_chain()

chain = load_chain()

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