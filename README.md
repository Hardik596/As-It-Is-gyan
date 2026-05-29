# 🕉️ As-It-Is Gyan

> A RAG-powered chatbot that answers questions using the wisdom of the **Bhagavad Gita As It Is** by Srila Prabhupada.

Ask anything — the AI retrieves the relevant passages from the scripture and answers faithfully, citing page numbers. If the answer isn't in the text, it says so.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | `mistral-small-latest` |
| Embeddings | `mistral-embed` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Backend | Flask + Gunicorn |

---

## Getting Started

**1. Clone & install**
```bash
git clone https://github.com/Hardik596/As-It-Is-gyan.git
cd As-It-Is-gyan
pip install -r requirements.txt
```

**2. Add your API key** — create a `.env` file:
```
MISTRAL_API_KEY=your_key_here
SECRET_KEY=any_secret_string
```

**3. Run**
```bash
python app.py
# or for production:
gunicorn app:app
```

Visit `http://localhost:5000`

> To rebuild the vector DB from scratch: `python create_db.py` (requires the PDF)

---

## Project Structure

```
├── app.py           # Flask backend & lazy RAG pipeline
├── create_db.py     # One-time script to build ChromaDB from PDF
├── main.py          # CLI / testing entry point
├── Gita-db/         # Pre-built ChromaDB vector store
└── requirements.txt
```

---

*First GenAI project — ancient wisdom meets modern retrieval-augmented generation.*
