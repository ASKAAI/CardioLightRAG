# retrieval/faiss_indexer.py
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import BIOASQ_PATH, DATA_PROCESSED_EMBEDDINGS_DIR, BIOBERT_MODEL

def create_faiss_index():
    """Create FAISS index from BioASQ data"""
    print("Processing BioASQ data...")
    
    # Load BioASQ data
    with open(BIOASQ_PATH) as f:
        data = json.load(f)
    
    # Extract questions and answers
    documents = []
    for item in data['questions']:
        question = item['body']
        exact_answer = item.get('exact_answer', [])
        if isinstance(exact_answer, list):
            exact_answer = " ".join(exact_answer)
        
        # Create document text
        doc_text = f"Question: {question}\nAnswer: {exact_answer}"
        documents.append(doc_text)
    
    # Initialize BioBERT model
    model = SentenceTransformer(BIOBERT_MODEL)
    
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
    
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    
    # Save index and documents
    os.makedirs(DATA_PROCESSED_EMBEDDINGS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_index.faiss'))
    
    with open(os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_documents.json'), 'w') as f:
        json.dump(documents, f)
    
    print(f"FAISS index created with {len(documents)} documents")

if __name__ == "__main__":
    create_faiss_index()