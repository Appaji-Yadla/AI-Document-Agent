import os
import streamlit as st
from pathlib import Path
import hashlib

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Constants
PERSIST_DIR = "chroma_db"
SUPPORTED_EXTS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx"
}

# Load API Key from Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Helper: Hashing file for duplicate detection
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Main document processing function
def process_document(file_path):
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        print(f"❌ Unsupported file type: {file_path.name}")
        return

    file_hash = get_file_hash(file_path)
    meta_path = Path(PERSIST_DIR) / f"{file_hash}.meta"
    if meta_path.exists():
        print(f"⚠️ Skipping already processed file: {file_path.name}")
        return

    print(f"📄 Processing file: {file_path.name}")
    loader = UnstructuredFileLoader(str(file_path), file_type=SUPPORTED_EXTS[ext])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        client_settings=Settings(anonymized_telemetry=False)

    )

    # Mark file as processed
    with open(meta_path, "w") as f:
        f.write("processed")

    print(f"✅ {file_path.name} processed and stored in vector DB.")
