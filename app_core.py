import os
from dotenv import load_dotenv

# Load environment variables from .env (if present). This allows OPENAI_API_KEY
# to be set in a local .env file during development.
load_dotenv()

# NOTE: LangChain is moving community-backed modules into `langchain_community`.
# The project currently uses the top-level imports which emit deprecation warnings.
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_llm():
    # Support either OPENAI_API_KEY (standard) or API_KEY (legacy in this repo)
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not openai_key:
        # Avoid raising a low-level validation error from pydantic — make it explicit
        raise RuntimeError(
            "Missing OpenAI API key. Set the environment variable OPENAI_API_KEY or API_KEY."
        )

    return ChatOpenAI(
        base_url=os.getenv("BASE_URL", None),
        model=os.getenv("MODEL", "google/gemini-2.0-flash-exp:free"),
        temperature=float(os.getenv("TEMPERATURE", 0.0)),
        max_retries=int(os.getenv("MAX_RETRIES", 3)),
        openai_api_key=openai_key,
    )

def load_faiss_index(faiss_index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # WARNING: FAISS.load_local performs pickle deserialization. Pickle files can be
    # tampered with to execute arbitrary code. Only set allow_dangerous_deserialization
    # to True if you trust the FAISS index file (for example, it was created locally).
    return FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )   
    
def answer_query(faiss_index_path, query):
    llm = get_llm()
    vector_store = load_faiss_index(faiss_index_path)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.run(query)
    return response