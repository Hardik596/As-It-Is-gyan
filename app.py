"""
app.py — Flask backend for As-It-Is Gyan
Bridges the browser UI with the existing RAG pipeline (main.py logic)
"""

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = "as-it-is-gyan-secret-key"  # For session management

# ─────────────────────────────────────────────
# RAG PIPELINE SETUP
# Initialised once at startup so every request
# reuses the same in-memory objects (faster).
# ─────────────────────────────────────────────

# Embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed")

# Load the persisted ChromaDB vector store
vectorstore = Chroma(
    persist_directory="Gita-db",
    embedding_function=embeddings
)

# MMR retriever: balances relevance + diversity
# k=5 final docs, fetch_k=20 candidates, lambda=0.5 for balance
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# Mistral LLM — the reasoning layer
llm = ChatMistralAI(model="mistral-small-2506")

# System prompt — guides the LLM to be a Gita counsellor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are As-It-Is Gyan, a compassionate and wise AI guide rooted in the teachings of the Bhagavad Gita As It Is by Srila Prabhupada.
     Help the seeker with their situation using the wisdom of the Gita. 
     Strictly use only the context retrieved from the sacred text. If you don't know the answer from the context, say so humbly.
     Never fabricate verses or teachings. Be concise, warm, and spiritually grounded in your tone.
     When relevant, mention the chapter and verse number from the context."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# In-memory chat history store keyed by session ID
# (for a production app, use Redis or a database)
chat_histories = {}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat page"""
    # Assign a unique session ID to each visitor
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: { "message": "<user query>" }
    Returns: { "response": "<AI answer>", "sources": [...] }
    """
    data = request.get_json()
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", "default")

    # Retrieve relevant chunks from the Gita vector store
    retrieved_docs = retriever.invoke(user_query)

    # Build context string from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build source references (page numbers) for citation
    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata
        page = meta.get("page", None)
        if page is not None:
            sources.append(f"Page {page + 1}")  # 0-indexed → 1-indexed
    sources = list(dict.fromkeys(sources))  # Deduplicate while preserving order

    # Invoke the LLM with context + question
    final_prompt = prompt.invoke({"context": context, "question": user_query})
    response = llm.invoke(final_prompt)

    # Store in chat history
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append({
        "user": user_query,
        "ai": response.content
    })

    return jsonify({
        "response": response.content,
        "sources": sources
    })


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear the chat history for the current session"""
    session_id = session.get("session_id", "default")
    chat_histories.pop(session_id, None)
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)