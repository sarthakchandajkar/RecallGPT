# test_rag.py
from rag_pipeline import load_retrieval_qa

qa = load_retrieval_qa()

while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("Answer:", answer)
