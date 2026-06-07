# 🏢 Enterprise RAG AI

An enterprise-grade Retrieval-Augmented Generation (RAG) application built to ingest corporate documents, store them in a vectorized database, and provide highly accurate, contextual answers using LLMs.

---

## 📌 About the Project

This repository contains a clean, production-ready RAG implementation designed to handle custom data ingestion and semantic search queries through an intuitive web interface. 

### ✨ Key Features
* **📄 Custom Data Ingestion:** Process and chunk raw text or documents from the `data/` directory.
* **⚡ Vector Storage:** High-performance vector embeddings stored and queried locally via **ChromaDB**.
* **🤖 Streamlit Interface:** A clean, responsive UI built with Streamlit for seamless user interaction and real-time chat responses.
* **🔒 Privacy Focused:** Designed to run without exposing sensitive corporate credentials or internal data.

---

## 📂 Project Structure

```text
├── chroma_db/          # Persistent vector database store
├── data/               # Source documents/text files for ingestion
├── app.py              # Main application configuration
├── ingest.py           # Script to chunk, embed, and load data into ChromaDB
├── requirements.txt    # Python dependencies
└── streamlit_app.py    # Streamlit UI definition and execution

