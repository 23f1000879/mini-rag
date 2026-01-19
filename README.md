Mini RAG Application 🚀

A simple Retrieval-Augmented Generation (RAG) application built with Flask (Python) and HTML frontend, capable of indexing documents and answering questions with inline citations.

Live Preview: - https://mini-rag-hrpe.onrender.com/

🧠 What This Project Does

Indexes text documents into a vector database

Retrieves relevant chunks for a user query

Generates grounded answers using an LLM

Shows inline citations like [1], [2]

Handles no-answer cases gracefully

🏗️ Tech Stack

Backend: Python + Flask

Frontend: HTML + Bootstrap (served via Flask)

Vector DB: Pinecone

LLM: Groq (LLaMA-3)

Reranker: Cohere

🚀 Quick Start
1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Set Environment Variables
Create a .env file:
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
COHERE_API_KEY=your-cohere-api-key

3️⃣ Start app
python app.py


📌 Index Documents

Upload a .txt file or paste text

Click Index Document

Wait for success message ✅

❓ Ask Questions

Enter your question

Select number of sources (default: 5)

Click Get Answer


View answer with citations 📚


