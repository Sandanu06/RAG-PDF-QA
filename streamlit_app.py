import streamlit as st
import os
import time
from ingest_and_index import ingest_pdf_to_faiss
from app_core import answer_query

st.title("PDF Question Answering with RAG")

# Ensure directories exist for uploads and indexes
os.makedirs("uploads", exist_ok=True)
os.makedirs("faiss_indices", exist_ok=True)

faiss_index_path = None
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save uploaded file to disk because the loader expects a file path
    upload_path = os.path.join("uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create a unique FAISS index path per upload
    base_name = os.path.splitext(uploaded_file.name)[0]
    faiss_index_path = os.path.join("faiss_indices", f"{base_name}_{int(time.time())}")

    # Ingest and index the saved PDF
    ingest_pdf_to_faiss(upload_path, faiss_index_path)
    st.success("PDF ingested and indexed successfully.")

query = st.text_input("Ask a question about the PDF:")
if query:
    if not faiss_index_path:
        st.error("Please upload and index a PDF before asking a question.")
    else:
        try:
            response = answer_query(faiss_index_path, query)
            st.write("Response:", response)
        except Exception as e:
            # Show a friendly error to the user
            st.error(f"Error while answering the query: {e}")
            # Also log to console for debugging
            st.write("(See server logs for full traceback)")