"""
app.py — Flask backend for As-It-Is Gyan
Fixed for Render free tier: lazy loading + gunicorn timeout safe
"""

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "as-it-is-gyan-secret-key")

# ─────────────────────────────────────────────────────────────────
# LAZY INITIALIZATION
# On Render's free tier, loading ChromaDB + MistralAI embeddings
# at startup consumes too much RAM and causes worker timeouts.
# Instead we initialize once on the FIRST request and reuse after.
# ─────────────────────────────────────────────────────────────────

_retriever = None
_llm = None
_prompt = None


def get_rag_components():
    """
    Initialize and cache RAG pipeline components.
    Called once on first /chat request, reused for all subsequent ones.
    """
    global _retriever, _llm, _prompt

    if _retriever is not None:
        return _retriever, _llm, _prompt  # Already initialized

    from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    print("Initializing RAG components on first request...")

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory="Gita-db",
        embedding_function=embeddings
    )

    _retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )

    _llm = ChatMistralAI(model="mistral-small-latest")

    _prompt = ChatPromptTemplate.from_messages([
        ("system", """You are As-It-Is Gyan, a compassionate and wise AI guide rooted in the teachings 
         of the Bhagavad Gita As It Is by Srila Prabhupada.
         Help the seeker using only the context retrieved from the sacred text. 
         If the answer is not in the context, say so humbly — never fabricate.
         Be concise, warm, and spiritually grounded. Mention chapter/verse when available."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    print("RAG components ready.")
    return _retriever, _llm, _prompt


chat_histories = {}


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Empty message"}), 400

    try:
        retriever, llm, prompt = get_rag_components()

        retrieved_docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        sources = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"Page {page + 1}")
        sources = list(dict.fromkeys(sources))

        final_prompt = prompt.invoke({"context": context, "question": user_query})
        response = llm.invoke(final_prompt)

        session_id = session.get("session_id", "default")
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({"user": user_query, "ai": response.content})

        return jsonify({"response": response.content, "sources": sources})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_history():
    session_id = session.get("session_id", "default")
    chat_histories.pop(session_id, None)
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
