import streamlit as st
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.abspath(os.path.join(current_dir, "..", "..", "backend"))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from backend.query import ask
from backend.ingest_main import ingest_data_main

st.set_page_config(page_title="📄 Main Document", layout="wide")
st.title("📄 Main Document")

# 🔑 Sidebar uploader for Main PDFs
st.sidebar.title("Upload Main PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more files",
    type=["pdf", "png", "jpg", "jpeg"],   
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data_main", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("data_main", file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success("✅ Files uploaded. Ready to look into the data .")

    if st.sidebar.button("Run Looking into the data"):
        ingest_data_main()   # calls your ingestion logic (saves embeddings + full_text.txt)
        st.sidebar.success("✅ Looking the data complete!")

query = st.text_input("Ask your question about the main document:")

if query:
    answer, reference = ask(query , db_path="chroma_db_main")
    st.write(f"**Answer:** {answer}")
    if reference:
        st.write("**References:**")
        for r in reference:
            st.write(f"- {r}")
