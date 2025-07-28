import streamlit as st
from load_document import load_file, split_documents, create_vector_store
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load API Key
api_key = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="AI Document Agent", layout="wide")
st.title("📄 AI Document Q&A Agent")

uploaded_file = st.file_uploader("Upload a document (PDF, Word, Excel, PowerPoint)", type=["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt"])

if uploaded_file:
    # Save uploaded file to temp
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and split documents
    docs = load_file(file_path)
    splits = split_documents(docs)
    vector_store = create_vector_store(splits, api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Set up LLM
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant for answering questions from company documents.

        Context: {context}

        Question: {question}
        """
    )
    chain = prompt_template | model | StrOutputParser()

    # User Query
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Generating answer..."):
            context_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            response = chain.invoke({"context": context, "question": question})
            st.success("Answer:")
            st.write(response)
