import os
import streamlit as st
import hashlib
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants testing
PERSIST_DIR = "chroma_db"
SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit UI
st.title("📄 AI Document Agent")
uploaded_file = st.file_uploader("Upload a document (PDF, Word, PPT, Excel)", type=["pdf", "docx", "pptx", "xlsx"])
user_input = st.text_area("💬 Ask your question below:")

# Save and hash uploaded file
def save_file(file):
    file_path = Path("uploaded_files") / file.name
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Optional: Call external script to process file (load_document.py logic)
def process_new_file(file_path):
    from load_document import process_document
    process_document(file_path)

# Process uploaded file
if uploaded_file:
    file_path = save_file(uploaded_file)
    file_hash = get_file_hash(file_path)
    meta_path = Path(PERSIST_DIR) / f"{file_hash}.meta"

    if not meta_path.exists():
        with st.spinner("Processing and embedding document..."):
            process_new_file(file_path)
        st.success("✅ File embedded and stored.")
    #else:
    #    st.info("ℹ️ This file is already processed.")

# Load Vector Store
vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Prompt Template
template = """
You are a helpful AI assistant trained on specific document or content uploaded by user.
Only answer questions based on the uploaded document. If the question is unrelated to those topics, politely respond that you cannot assist.

Use the context below to answer the question.

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

if user_input:
    with st.spinner("Generating response..."):
        result = rag_chain.invoke(user_input)
        st.markdown("### 🤖 AI Response:")
        st.write(result)