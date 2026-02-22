import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index_name="rag-index",
    embedding=embeddings
)

retriever = vector_store.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

query = input("Haz una pregunta: ")

docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
Responde la pregunta usando únicamente el siguiente contexto.

Contexto:
{context}

Pregunta:
{query}
"""

response = llm.invoke(prompt)

print("\nRespuesta:\n")
print(response.content)