# RAG-PDF-QA

A minimal Retrieval-Augmented Generation (RAG) demo: upload a PDF, index it with FAISS, and ask questions using a LangChain-powered retrieval + LLM pipeline served via Streamlit.

This repo provides a compact, practical starting point for building a PDF QA app locally.

---

## Highlights

- Upload a PDF in the browser (Streamlit).
- Ingest the PDF into chunks and create embeddings with HuggingFace sentence-transformers.
- Store vectors in a local FAISS index and use a retrieval-augmented QA chain to answer queries.
- Minimal, focused code you can extend for production use.

---

## Files

- `streamlit_app.py` — Streamlit front-end. Saves uploads, creates per-upload FAISS indices, and displays responses.
- `ingest_and_index.py` — Ingests PDFs, splits text, creates embeddings, and saves a FAISS index.
- `app_core.py` — Core runtime: loads FAISS index, creates the LLM, and runs the RetrievalQA chain.
- `requirements.txt` — Python dependencies (use the included virtualenv or create your own).
- `.env` — Local environment variables (not checked into git).

---

## Quick start (local)

1. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Add your OpenAI-compatible API key to a `.env` file in the repo root (example included):

```properties
# .env
OPENAI_API_KEY=sk-...
BASE_URL=https://openrouter.ai/api/v1   # optional; used if routing to an alternative OpenAI-compatible endpoint
```

4. Run the app:

```bash
./venv/bin/streamlit run streamlit_app.py
```

5. Open the URL printed by Streamlit (usually `http://localhost:8501`) and upload a PDF.

---

## Environment variables

- `OPENAI_API_KEY` (required) — API key for the LLM provider.
- `API_KEY` — legacy alias also supported.
- `BASE_URL` — optional custom base URL for OpenAI-compatible endpoints.
- `MODEL` — (optional) model identifier. Defaults to a configured model in `app_core.py`.
- `TEMPERATURE`, `MAX_RETRIES` — tuning parameters.

The app calls `load_dotenv()` so variables in `.env` are loaded automatically during development.

---

## Security notes (important)

- FAISS indexes are saved and loaded using Python pickle under the hood. Loading a tampered pickle file can execute arbitrary code on your machine.
  - The repo sets `allow_dangerous_deserialization=True` when loading a FAISS index for convenience during development.
  - ONLY use this for indices you created and trust. If you ever download an index from an untrusted source, do NOT enable unsafe deserialization.

- Never commit `.env` or your API keys to source control. `.gitignore` is configured to exclude the `.env` file.

---

## Next steps & suggestions

- Replace deprecated LangChain imports with `langchain_community` equivalents to avoid runtime deprecation warnings.
- Pin package versions in `requirements.txt` to make the environment reproducible.
- Add unit tests and a small smoke test that ingests a tiny PDF and runs a simple query.
- Consider swapping in local LLMs or private endpoints if you prefer not to use an external API.

---

## Troubleshooting

- If you see `ModuleNotFoundError: langchain_community` — run `pip install -r requirements.txt`.
- If the app complains about a missing OpenAI key: ensure `OPENAI_API_KEY` or `API_KEY` is set in your environment or `.env`.
- If you see deprecation warnings from LangChain, consider migrating imports per the messages.

---

## License

This project contains example code for experimentation. Review and adapt for your needs.

---

Happy building! If you'd like, I can migrate imports to `langchain_community` now and pin dependency versions in `requirements.txt`.