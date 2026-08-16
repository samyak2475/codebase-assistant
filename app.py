import streamlit as st
from src.query import get_qa_chain
from src.indexer import create_vectorstore

st.set_page_config(page_title="Local Codebase Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Local Codebase Assistant")
st.caption("Fully local • Powered by Ollama • Zero cloud")

# Sidebar
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
                    st.success("Indexing completed!")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.header("Query Mode")
    mode = st.radio(
        "Choose mode:",
        ["General Question", "Explain Function/Class", "Architecture Overview", "Suggest Improvements"],
        index=0
    )
    
    st.markdown("---")
    st.header("Filter (Optional)")
    filter_type = st.selectbox(
        "Focus only on:",
        ["All", "Functions only", "Classes only"],
        index=0
    )
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    - 100% private
    - Runs fully on your machine
    - Uses small local models
    """)
    st.code("qwen2.5-coder:3b\nnomic-embed-text")

# Convert filter selection
filter_map = {
    "All": None,
    "Functions only": "function",
    "Classes only": "class"
}
selected_filter = filter_map[filter_type]

# Load chain with filter
@st.cache_resource
def load_chain(filter_type):
    return get_qa_chain(filter_type=filter_type)

try:
    chain = load_chain(selected_filter)
except Exception:
    st.info("No codebase indexed yet. Please index a folder using the sidebar.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the codebase..."):
    
    # Build prompt based on mode
    if mode == "Explain Function/Class":
        full_prompt = f"Explain the following function or class in detail, including what it does, parameters, and how it works:\n\n{prompt}"
    elif mode == "Architecture Overview":
        full_prompt = f"Give a clear high-level architecture overview of this project. Focus on main components and how they connect. Question: {prompt}"
    elif mode == "Suggest Improvements":
        full_prompt = f"Suggest concrete improvements, potential bugs, or code smells for this codebase. Be specific. Question: {prompt}"
    else:
        full_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(full_prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})