from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def ingest_pdf_to_faiss(pdf_path, faiss_index_path):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings using a HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store from the documents and embeddings
    db = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to disk
    db.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")
    return faiss_index_path