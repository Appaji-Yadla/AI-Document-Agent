import os
import streamlit as st
import hashlib
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure async loop (for Streamlit compatibility)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Constants
PERSIST_DIR = "chroma_db"
SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Load API Key securely
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# UI Layout
st.title("📄 AI Document Agent")
uploaded_file = st.file_uploader("Upload a document (PDF, Word, PPT, Excel)", type=list(SUPPORTED_EXTS))
user_input = st.text_area("💬 Ask your question below:")

# Save uploaded file to disk
def save_file(file):
    file_path = Path("uploaded_files") / file.name
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Import and call loader
def process_new_file(file_path):
    from load_document import process_document
    process_document(file_path)

# File handling
if uploaded_file:
    file_path = save_file(uploaded_file)
    file_hash = get_file_hash(file_path)
    meta_path = Path(PERSIST_DIR) / f"{file_hash}.meta"

    if not meta_path.exists():
        with st.spinner("Processing and embedding document..."):
            process_new_file(file_path)
        st.success("✅ File embedded and stored.")
    else:
        st.info("ℹ️ This file was already processed. Using existing data.")

# Load Chroma vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    client_settings=Settings(anonymized_telemetry=False),
    #in_memory=True  # <--- This makes it compatible with Streamlit Cloud
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Prompt Template for RAG
template = """
You are a helpful AI assistant trained on specific documents uploaded by the user.
Only answer questions based on the uploaded document. If the question is unrelated, respond politely that you cannot help.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
parser = StrOutputParser()

rag_chain = (
    {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Answer user query
if user_input:
    with st.spinner("Generating response..."):
        result = rag_chain.invoke(user_input)
        st.markdown("### 🤖 AI Response:")
        st.write(result)
