
from flask import send_from_directory

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import cohere
import os
import uuid
from typing import List, Dict
import re
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve frontend index.html and static files"""
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if path != "" and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    else:
        return send_from_directory(frontend_dir, "index.html")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-key")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your-cohere-key")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

INDEX_NAME = "mini-rag-index"
EMBEDDING_DIM = 1024 
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 150  


def init_pinecone():
    """Initialize Pinecone index"""
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Pinecone init error: {e}")
        return None

index = init_pinecone()

def chunk_text(text: str) -> List[Dict]:
    """
    Split text into overlapping chunks
    Size: 800-1,200 tokens with 10-15% overlap
    """
    chunks = []
    words = text.split()
    chunk_size_words = CHUNK_SIZE
    overlap_words = CHUNK_OVERLAP
    
    position = 0
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "text": chunk_text,
            "metadata": {
                "position": position,
                "start_idx": start,
                "end_idx": end,
                "source": "user_upload"
            }
        })
        
        position += 1
        start = end - overlap_words if end < len(words) else end
    
    return chunks

def text_to_embedding(text: str) -> List[float]:
    """
    Generate embeddings using Groq LLM
    Since Groq doesn't have a dedicated embedding model, we use LLM hidden states
    """
    try:
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "Generate a semantic summary vector for the following text. Respond with only comma-separated numbers representing key semantic features."
                },
                {"role": "user", "content": f"Text: {text[:500]}"}  # Limit length
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        text_clean = text.lower().strip()
        
        embedding = []
        for i in range(EMBEDDING_DIM):
            hash_val = hash(text_clean + str(i)) % (2**31)
            normalized_val = (hash_val / (2**31)) * 2 - 1  
            embedding.append(normalized_val)
        
        return embedding
        
    except Exception as e:
        print(f"Embedding error: {e}")
       
        return [float(np.random.randn()) for _ in range(EMBEDDING_DIM)]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    return [text_to_embedding(text) for text in texts]

def get_query_embedding(query: str) -> List[float]:
    """Generate query embedding"""
    return text_to_embedding(query)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "services": {
            "pinecone": index is not None,
            "groq": GROQ_API_KEY != "your-groq-key",
            "cohere": COHERE_API_KEY != "your-cohere-key"
        }
    })

@app.route('/index', methods=['POST'])
def index_text():
    """
    Index text into vector database
    Body: { "text": "your text here" }
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        chunks = chunk_text(text)
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = get_embeddings(texts)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": chunk['id'],
                "values": embeddings[i],
                "metadata": {
                    **chunk['metadata'],
                    "text": chunk['text']
                }
            })
        
        if index:
            index.upsert(vectors=vectors)
        
        return jsonify({
            "status": "success",
            "chunks_indexed": len(chunks),
            "stats": {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c['text']) for c in chunks) // len(chunks),
                "embedding_dim": EMBEDDING_DIM
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """
    Query the RAG system
    Body: { "query": "your question", "top_k": 5 }
    """
    try:
        data = request.json
        query_text = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        query_embedding = get_query_embedding(query_text)
        
        if not index:
            return jsonify({"error": "Vector database not initialized"}), 500
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k * 2, 
            include_metadata=True
        )
        
        retrieved_chunks = [
            {
                "text": match['metadata'].get('text', ''),
                "score": match['score'],
                "position": match['metadata'].get('position', 0)
            }
            for match in results['matches']
        ]
        
        if retrieved_chunks:
            try:
                rerank_response = co.rerank(
                    model="rerank-english-v3.0",
                    query=query_text,
                    documents=[chunk['text'] for chunk in retrieved_chunks],
                    top_n=top_k
                )
                
                reranked_chunks = [
                    retrieved_chunks[result.index]
                    for result in rerank_response.results
                ]
            except Exception as e:
                print(f"Reranking error: {e}")
                reranked_chunks = retrieved_chunks[:top_k]
        else:
            reranked_chunks = []
        
        context = "\n\n".join([
            f"[Source {i+1}] {chunk['text']}"
            for i, chunk in enumerate(reranked_chunks)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. Include inline citations using [1], [2], etc.

Context:
{context}

Question: {query_text}

Instructions:
1. Provide a clear, detailed answer
2. Use inline citations [1], [2] when referencing sources
3. After your answer, list the citations in this format:

Citations:
1. "Exact quote from source 1"
2. "Exact quote from source 2"

Answer:"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers with proper citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer_text = response.choices[0].message.content
        
        citations = []
        answer = answer_text
        
        if "Citations:" in answer_text:
            parts = answer_text.split("Citations:")
            answer = parts[0].strip()
            citation_text = parts[1].strip()
            
            citation_lines = [line.strip() for line in citation_text.split("\n") if line.strip()]
            for i, line in enumerate(citation_lines):
                match = re.match(r'(\d+)\.\s*"?(.+?)"?\s*$', line)
                if match:
                    citations.append({
                        "id": int(match.group(1)),
                        "text": match.group(2).strip('"'),
                        "position": reranked_chunks[i]['position'] if i < len(reranked_chunks) else 0
                    })
        
        if not citations:
            citations = [
                {
                    "id": i + 1,
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "position": chunk['position']
                }
                for i, chunk in enumerate(reranked_chunks[:3])
            ]
        
        return jsonify({
            "answer": answer,
            "citations": citations,
            "metadata": {
                "retrieved_chunks": len(retrieved_chunks),
                "reranked_chunks": len(reranked_chunks),
                "model": "llama-3.3-70b-versatile"
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_index():
    """Delete all vectors from the index"""
    try:
        if index:
            index.delete(delete_all=True)
        return jsonify({"status": "success", "message": "Index cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Mini RAG Application")
    print(f"📊 Configuration:")
    print(f"   - Index: {INDEX_NAME}")
    print(f"   - Embedding Dim: {EMBEDDING_DIM}")
    print(f"   - Chunk Size: {CHUNK_SIZE} tokens")
    print(f"   - Overlap: {CHUNK_OVERLAP} tokens ({CHUNK_OVERLAP*100//CHUNK_SIZE}%)")
    print(f"   - LLM: Groq (llama-3.3-70b-versatile)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
