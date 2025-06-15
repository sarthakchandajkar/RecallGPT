import streamlit as st
import os
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import shutil
from parsers import aggregate_notes
import pandas as pd

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="RecallGPT", page_icon="🧠", layout="wide")

# ------------------- Custom Styling -------------------
st.markdown("""
    <style>
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h3, p {
            text-align: center;
        }

        div[data-testid="stTextInput"] > div:first-child {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stTextInput"] input {
            width: 100%;
            max-width: 600px;
            text-align: center;
        }

        div.stButton {
            display: flex;
            justify-content: center;
        }

        .stDataFrameContainer {
            margin-top: 1.5rem;
        }

        .message {
            padding: 12px 20px;
            border-radius: 12px;
            margin-bottom: 10px;
            line-height: 1.5;
            font-size: 16px;
        }

        .user-msg {
            background-color: #dbeafe;
            color: #111827;
            text-align: right;
        }

        .ai-msg {
            background-color: #f3f4f6;
            color: #111827;
            text-align: left;
        }

        .clear-button-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
mode = st.sidebar.radio("📌 Choose Assistant Mode:", ["Chat with Notes", "Tasks", "Expenses"])

uploaded_files = st.sidebar.file_uploader("📁 Upload `.md` Notes", type=["md"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs("notes", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("notes", file.name), "wb") as f:
            f.write(file.getvalue())
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

# ------------------- Retrain Button -------------------
def retrain_vector_db():
    st.session_state['retrained'] = False
    loader = DirectoryLoader(
        path="./notes",
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    st.session_state['retrained'] = True

if st.sidebar.button("🔄 Retrain Vector DB"):
    with st.spinner("Rebuilding vector database..."):
        retrain_vector_db()
    st.sidebar.success("✅ Vector DB updated!")

# ------------------- Load Chain -------------------
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    llm = Ollama(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

qa_chain = load_chain()

# ------------------- Chat UI -------------------
if mode == "Chat with Notes":
    st.title("🧠 RecallGPT")
    st.write("Ask questions based on your uploaded notes!")

    st.markdown("<div class='clear-button-container'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation"):
        st.session_state["history"] = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state["history"] = []

    query = st.text_input("🔎 Ask your assistant:", placeholder="e.g., What did I do on 2024-10-02?")
    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.session_state["history"].append((query, response))

    if st.session_state["history"]:
        st.divider()
        for q, a in reversed(st.session_state["history"]):
            st.markdown(f"<div class='message user-msg'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='message ai-msg'><b>RecallGPT:</b> {a}</div>", unsafe_allow_html=True)

# ------------------- Tasks -------------------
elif mode == "Tasks":
    st.title("✅ Task Overview")
    all_tasks, _ = aggregate_notes("notes")
    df = pd.DataFrame(all_tasks)

    if df.empty:
        st.info("No tasks found.")
    else:
        status_filter = st.selectbox("Filter by status", ["all", "todo", "done"])
        date_filter = st.text_input("Filter by date (YYYY-MM-DD)", "")

        filtered = df.copy()
        if status_filter != "all":
            filtered = filtered[filtered["status"] == status_filter]
        if date_filter:
            filtered = filtered[filtered["date"] == date_filter]

        st.dataframe(filtered)

# ------------------- Expenses -------------------
elif mode == "Expenses":
    st.title("💸 Expense Tracker")
    _, all_expenses = aggregate_notes("notes")
    df = pd.DataFrame(all_expenses)

    if df.empty:
        st.info("No expenses found.")
    else:
        date_filter = st.text_input("Filter by date (YYYY-MM-DD or YYYY-MM)", "")
        category_filter = st.text_input("Filter by category (optional)", "").lower()

        filtered = df.copy()
        if date_filter:
            filtered = filtered[filtered["date"].str.startswith(date_filter)]
        if category_filter:
            filtered = filtered[filtered["category"].str.lower().str.contains(category_filter)]

        st.dataframe(filtered)
        st.markdown(f"### 💰 Total: ₹{filtered['amount'].sum():,.2f}")
