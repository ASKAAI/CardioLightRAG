import os
import sys
import json
import numpy as np

# Add parent directory to Python path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers import SentenceTransformer
from config import BIOBERT_MODEL, DATA_PROCESSED_EMBEDDINGS_DIR, BIOASQ_PATH

class EmbeddingGenerator:
    def __init__(self, model_name=BIOBERT_MODEL):
        self.model = SentenceTransformer(model_name)
    
    def generate_bioasq_embeddings(self):
        """Generate embeddings for BioASQ dataset"""
        print("Loading BioASQ data...")
        with open(BIOASQ_PATH) as f:
            data = json.load(f)
        
        documents = []
        for item in data['questions']:
            doc_text = f"Question: {item['body']}\n"
            if 'exact_answer' in item:
                if isinstance(item['exact_answer'], list):
                    doc_text += "Answer: " + " ".join(item['exact_answer'])
                else:
                    doc_text += "Answer: " + item['exact_answer']
            documents.append(doc_text)
        
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=32)
        
        print("Saving embeddings...")
        os.makedirs(DATA_PROCESSED_EMBEDDINGS_DIR, exist_ok=True)
        np.save(os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_embeddings.npy'), embeddings)
        with open(os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_documents.json'), 'w') as f:
            json.dump(documents, f)
        
        return embeddings

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.generate_bioasq_embeddings()