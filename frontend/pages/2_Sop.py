import streamlit as st
import json
from backend.ingest import ingest_data_main
from backend.query import sop_check

# --- Load SOP queries from JSON ---
with open("sop_queries.json", "r") as f:
    sop_data = json.load(f)

st.sidebar.title("SOP Assistant")
category = st.sidebar.selectbox("Select Category", list(sop_data.keys()))
selected_query_obj = st.sidebar.selectbox(
    "Select SOP Query", sop_data[category], format_func=lambda x: x["query"]
)

uploaded_pdf = st.sidebar.file_uploader("Upload PDF for SOP check", type=["pdf"])

# --- Run button ---
if st.sidebar.button("Run SOP Check"):
    if uploaded_pdf is not None:
        # Save uploaded file to data_main/
        with open(f"data_main/{uploaded_pdf.name}", "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.info("Running ingestion...")
        ingest_data_main()

        st.info("Running SOP checks...")
        # Pass selected query if chosen, otherwise run all
        if selected_query_obj:
            queries = [selected_query_obj["query"]]
        else:
            queries = [q["query"] for q in sop_data[category]]

        sop_results = sop_check(queries, db_path="chroma_db_main", use_qwen=True)

        st.subheader("Results")
        for q, res in sop_results.items():
            color = "✅" if res["matched"] else "❌"
            st.write(f"{color} **Q:** {q}")
            st.write(f"Matched: {res['matched']}")
            st.write(f"Reference Items: {', '.join(res.get('reference_items', []))}")
            st.write(f"Checking Items: {', '.join(res.get('checking_items', []))}")
            st.write(f"Missing in Checking: {', '.join(res.get('missing_in_check', []))}")
            st.write(f"Extra in Checking: {', '.join(res.get('extra_in_check', []))}")
            st.write("---")
    else:
        st.warning("Please upload a PDF before running the SOP check.")
