import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Streamlit UI setup
st.title("Conversational RAG with PDF uploads and Chat History")
st.sidebar.title("Upload PDFs and Chat with Their Content")

# Persistent vectorstore directory
PERSIST_DIRECTORY = "./chroma_db"

# Get API Key from secrets
if "groq" in st.secrets:
    api_key = st.secrets["groq"]["API_KEY"]
else:
    st.warning("Please configure API key in Streamlit secrets")
    st.stop()

model = st.sidebar.selectbox("Select LLM model", (
    "qwen-2.5-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it",
    "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "Gemma2-9b-It"
))
llm = ChatGroq(groq_api_key=api_key, model_name=model)

# Session ID
session_id = st.sidebar.text_input("Session ID", value="default_session")

# State for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    retriever = vectorstore.as_retriever()

    # Setup Conversational RAG Chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # Get chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    user_input = st.text_input("Ask a question about the uploaded documents:")
    if user_input:
        chat_history = get_session_history(session_id)
        response = rag_chain({
            "question": user_input,
            "chat_history": [(msg.content if msg.type == "human" else "Assistant: " + msg.content)
                             for msg in chat_history.messages]
        })

        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(response["answer"])

        st.markdown(f"**Assistant:** {response['answer']}")
        st.markdown("---")
        st.markdown("**Chat History:**")
        for msg in chat_history.messages:
            st.write(f"{msg.type.capitalize()}: {msg.content}")
