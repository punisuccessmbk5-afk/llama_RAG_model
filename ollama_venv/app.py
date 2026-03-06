"""
Streamlit Frontend.
Handles UI only.
"""

import streamlit as st
import time

from backend.document_loader import load_documents
from backend.vector_store import load_or_create_index
from backend.embeddings import get_embedding
from backend.embedding_cache import load_cached_embeddings
from backend.rag_pipeline import query_rag


# -----------------------------
# Page Config + Styling
# -----------------------------
st.set_page_config(
    page_title="Enterprise RAG Tool",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #0e1117;}
.stButton>button {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Enterprise RAG Assistant")

# -----------------------------
# Session State
# -----------------------------
if "cache" not in st.session_state:
    st.session_state.cache = load_cached_embeddings()

if "index" not in st.session_state:
    st.session_state.index = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["xlsx", "pdf", "docx", "png", "jpg"],
        accept_multiple_files=True
    )

# -----------------------------
# Main Chat Area
# -----------------------------
question = st.chat_input("Ask your question...")

if uploaded_files and question:

    with st.spinner("Processing documents..."):
        docs = load_documents(uploaded_files, question)

        embeddings = [
            get_embedding(doc.page_content, st.session_state.cache)
            for doc in docs
        ]

        st.session_state.index = load_or_create_index(embeddings)

    st.success("Documents Indexed Successfully!")

    answer_container = st.empty()

    def stream_update(text):
        answer_container.markdown(text)

    with st.spinner("Generating answer..."):
        answer, retrieved = query_rag(
            st.session_state.index,
            docs,
            question,
            st.session_state.cache,
            stream_callback=stream_update
        )

    st.success("Answer Generated!")

    st.subheader("📄 Retrieved Sources")
    for doc in retrieved:
        st.write(doc.metadata["source"])