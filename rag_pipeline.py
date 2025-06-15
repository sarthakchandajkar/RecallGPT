# rag_pipeline.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

import os

# ✅ Step 1: Load all notes from markdown files
loader = DirectoryLoader(
    path="./notes",
    glob="**/*.md",
    loader_cls=TextLoader,   # <- Force use of simple loader
    show_progress=True
)
docs = loader.load()
print(f"🔍 Loaded {len(docs)} documents.")

# ✅ Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"🧩 Created {len(chunks)} chunks.")

# ✅ Step 3: Convert text to embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Optional: Remove old DB to avoid duplication
if os.path.exists("./chroma_db"):
    import shutil
    shutil.rmtree("./chroma_db")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vectordb.persist()

# ✅ Step 4: Load LLM from Ollama (make sure it's running)
llm = Ollama(model="llama3")  # Not "llama3.2" — use just "llama3" unless you're using a custom tag

# ✅ Step 5: Create RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# ✅ Step 6: Ask questions
print("\n✅ Assistant is ready! Type your questions (or type 'exit' to quit)\n")
while True:
    query = input("❓ You: ")
    if query.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    answer = qa_chain.run(query)
    print("🤖 Answer:", answer)
