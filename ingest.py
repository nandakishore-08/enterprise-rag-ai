import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
CHROMA_DB = "chroma_db"

documents = []

print("Loading PDFs...")

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_FOLDER, file)
        print(f"Reading: {file}")

        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print(f"\nLoaded {len(documents)} pages")

print("\nSplitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nCreating ChromaDB...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB
)

print("\n✅ ChromaDB created successfully!")
print("✅ Your RAG knowledge base is ready!")