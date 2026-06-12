import streamlit as st
import sys, os
import difflib

sys.path.append(os.path.abspath("backend"))

from backend.query import ask
from backend.ingest_main import ingest_data_main
from backend.ingest_compared import ingest_data_compared

st.set_page_config(page_title="📄 Compare Side by Side", layout="wide")
st.title("📄 Compare Main vs Compared Document")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar uploaders
st.sidebar.title("Upload Documents for Comparison")

# Upload Main Documents (PDF + Images)
uploaded_main = st.sidebar.file_uploader(
    "Upload Main Documents", type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True, key="main"
)

if uploaded_main:
    os.makedirs("data_main", exist_ok=True)
    for file in uploaded_main:
        save_path = os.path.join("data_main", file.name)
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                st.error(f"File {file.name} is open. Close it and try again.")
                continue
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success("✅ Main files uploaded.")
    if st.sidebar.button("Run looking into the Main data"):
        ingest_data_main()
        st.sidebar.success("✅ Looking into the Main data complete!")

# Upload Compared Documents (PDF + Images)
uploaded_compared = st.sidebar.file_uploader(
    "Upload Compared Documents", type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True, key="compared"
)

if uploaded_compared:
    os.makedirs("data_compared", exist_ok=True)
    for file in uploaded_compared:
        save_path = os.path.join("data_compared", file.name)
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                st.error(f"File {file.name} is open. Close it and try again.")
                continue
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success("✅ Compared files uploaded.")
    if st.sidebar.button("Run looking into the Compared data"):
        ingest_data_compared()
        st.sidebar.success("✅ Looking into the Compared data complete!")

# Query box
query = st.text_input("Enter your question (e.g. 'compare docs', 'compare efficiency', 'compare matching and not matching details'):")

if query:
    q_lower = query.strip().lower()

    # Case 1: Full comparison mode
    if q_lower in ["compare docs", "compare documents"]:
        try:
            with open("data_main/combined.txt", "r", encoding="utf-8") as f:
                main_text = f.read()
            with open("data_compared/combined.txt", "r", encoding="utf-8") as f:
                compared_text = f.read()
        except FileNotFoundError:
            st.error("⚠️ Please run looking into the data for both Main and Compared documents first.")
            main_text, compared_text = "", ""

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Main Document (Full Details)")
            st.text(main_text)
        with col2:
            st.subheader("Compared Document (Full Details)")
            st.text(compared_text)

        # Line-by-line comparison
        main_lines = main_text.splitlines()
        comp_lines = compared_text.splitlines()
        diff = difflib.ndiff(main_lines, comp_lines)

        st.subheader("Comparison Summary")
        for line in diff:
            if line.startswith("- "):
                st.markdown(f"**Main only:** {line[2:]}")
            elif line.startswith("+ "):
                st.markdown(f"**Compared only:** {line[2:]}")
            elif line.startswith("  "):
                st.markdown(f"**Matching:** {line[2:]}")

    # Case 2: Keyword-based comparison
    elif q_lower.startswith("compare "):
        keyword = q_lower.replace("compare ", "").strip()
        try:
            with open("data_main/combined.txt", "r", encoding="utf-8") as f:
                main_text = f.read()
            with open("data_compared/combined.txt", "r", encoding="utf-8") as f:
                compared_text = f.read()
        except FileNotFoundError:
            st.error("⚠️ Please run looking into the main data  for both Main and Compared documents first.")
            main_text, compared_text = "", ""

        main_lines = [line for line in main_text.splitlines() if keyword in line.lower()]
        comp_lines = [line for line in compared_text.splitlines() if keyword in line.lower()]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Main Document ({keyword})")
            st.text("\n".join(main_lines) if main_lines else "No match found.")
        with col2:
            st.subheader(f"Compared Document ({keyword})")
            st.text("\n".join(comp_lines) if comp_lines else "No match found.")

        diff = difflib.ndiff(main_lines, comp_lines)
        st.subheader("Comparison Summary")
        for line in diff:
            if line.startswith("- "):
                st.markdown(f"**Main only:** {line[2:]}")
            elif line.startswith("+ "):
                st.markdown(f"**Compared only:** {line[2:]}")
            elif line.startswith("  "):
                st.markdown(f"**Matching:** {line[2:]}")

    # Case 3: Query for Main only
    elif "main" in q_lower:
        answer_main, reference_main = ask(query, db_path="chroma_db_main")
        st.session_state.chat_history.append({
            "query": query,
            "main_answer": answer_main,
            "main_refs": reference_main,
            "comp_answer": None,
            "comp_refs": [],
        })

    # Case 4: Query for Compared only
    elif "compared" in q_lower:
        answer_compared, reference_compared = ask(query, db_path="chroma_db_compared")
        st.session_state.chat_history.append({
            "query": query,
            "main_answer": None,
            "main_refs": [],
            "comp_answer": answer_compared,
            "comp_refs": reference_compared,
        })

    # Case 5: Generic query → ask both
    else:
        answer_main, reference_main = ask(query, db_path="chroma_db_main")
        answer_compared, reference_compared = ask(query, db_path="chroma_db_compared")
        st.session_state.chat_history.append({
            "query": query,
            "main_answer": answer_main,
            "main_refs": reference_main,
            "comp_answer": answer_compared,
            "comp_refs": reference_compared,
        })

# Display chat history
for entry in st.session_state.chat_history:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Main Document")
        if entry["main_answer"]:
            st.markdown(f"**Answer:** {entry['main_answer']}")
            if entry["main_refs"]:
                st.markdown("**References:**")
                for r in entry["main_refs"]:
                    st.markdown(f"- {r}")
        else:
            st.info("No query for Main Document.")

    with col2:
        st.subheader("Compared Document")
        if entry["comp_answer"]:
            st.markdown(f"**Answer:** {entry['comp_answer']}")
            if entry["comp_refs"]:
                st.markdown("**References:**")
                for r in entry["comp_refs"]:
                    st.markdown(f"- {r}")
        else:
            st.info("No query for Compared Document.")
