import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader

load_dotenv()

# ---------------- PAGE ---------------- #
st.set_page_config(page_title="Enterprise RAG AI", layout="wide")

# ---------------- LOG FILE ---------------- #
LOG_FILE = "audit_logs.json"

def save_log(user, role, question, answer):
    entry = {
        "user": user,
        "role": role,
        "question": question,
        "answer": answer,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# ---------------- USERS ---------------- #
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LOGIN ---------------- #
def login():
    st.markdown("<h1 style='text-align:center;color:white;'>🔐 Login</h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- DB ---------------- #
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_db():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=get_embeddings()
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

embeddings = get_embeddings()
db = get_db()
llm = get_llm()

# ---------------- PDF UPLOAD ---------------- #
def process_pdf(file, role):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs = [
        Document(page_content=chunk, metadata={"role": role})
        for chunk in chunks
    ]

    db.add_documents(docs)
    db.persist()

# ---------------- UI STYLE ---------------- #
st.markdown("""
<style>

.stApp {
    background: #0b0f19;
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
    margin: 8px 0;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.12);
}

.stChatInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

section[data-testid="stSidebar"] {
    background-color: #0a0f1c;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.title("🤖 Enterprise RAG System")

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.write(f"👤 User: {st.session_state.username}")
    st.write(f"🔐 Role: {st.session_state.role}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("---")

    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload", type=["pdf"])
    assign_role = st.selectbox("Role", ["admin", "user"])

    if uploaded_file and st.button("Process"):
        process_pdf(uploaded_file, assign_role)
        st.success("Uploaded!")

    st.markdown("---")

    # ---------------- AUDIT LOGS ---------------- #
    if st.session_state.role == "admin":
        st.subheader("📊 Audit Logs")

        if st.button("Show Logs"):
            try:
                with open(LOG_FILE, "r") as f:
                    logs = json.load(f)

                st.json(logs[-10:])
            except:
                st.info("No logs found")

# ---------------- CHAT HISTORY ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT ---------------- #
prompt = st.chat_input("Ask from your documents...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    docs = db.similarity_search(prompt, k=10)

    role = st.session_state.role

    filtered_docs = []

    for d in docs:
        doc_role = d.metadata.get("role", "user")

        if role == "admin":
            filtered_docs.append(d)
        elif doc_role == "user":
            filtered_docs.append(d)

    if not filtered_docs:
        filtered_docs = docs[:3]

    context = "\n\n".join([d.page_content for d in filtered_docs])

    final_prompt = f"""
Use ONLY the context below:

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

    # ---------------- AUDIT LOG ---------------- #
    save_log(
        st.session_state.username,
        st.session_state.role,
        prompt,
        response.content
    )