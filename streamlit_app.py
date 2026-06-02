import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI STYLE ---------------- #
st.markdown("""
<style>

/* App background (clean dark SaaS look) */
.stApp {
    background: #0b0f19;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* Top Title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    color: #ffffff;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 20px;
}

/* Chat message container */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 10px;
    margin: 10px 0;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}

/* User message */
[data-testid="stChatMessage"]:nth-child(odd) {
    background: rgba(56,189,248,0.08);
}

/* Assistant message */
[data-testid="stChatMessage"]:nth-child(even) {
    background: rgba(34,211,238,0.06);
}

/* Input box */
.stChatInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #374151 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0f1c;
    border-right: 1px solid #1f2937;
}

/* Sidebar text */
.css-1d391kg {
    color: white;
}

/* Metric cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<div class='main-title'>🤖 Enterprise RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Chat with your documents using AI-powered retrieval</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("📊 Control Panel")

    st.markdown("### System Status")
    st.success("🟢 Online")

    st.markdown("### Stack")
    st.write("• LangChain")
    st.write("• Groq LLM")
    st.write("• ChromaDB")
    st.write("• RAG Pipeline")

    st.markdown("### Actions")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- INIT ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# ---------------- CHAT HISTORY ---------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- INPUT ---------------- #
prompt = st.chat_input("Ask anything from your documents...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    docs = db.similarity_search(prompt, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"""
Use only the context below.

Context:
{context}

Question:
{prompt}
"""

    response = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })