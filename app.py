import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI STYLE (WHITE + CLEAN) ----------------
st.markdown("""
<style>
.stApp {
    background: #ffffff;
    color: #111111;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #111111;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #555555;
    margin-bottom: 30px;
}

[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #f3f4f6;
}

[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #e5e7eb;
}

.stChatInput input {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🤖 Enterprise RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Chat with your documents using AI-powered retrieval</div>", unsafe_allow_html=True)

# ---------------- CHECK API KEY ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# ---------------- LOAD LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR DB ----------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
query = st.chat_input("Ask something from your documents...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )