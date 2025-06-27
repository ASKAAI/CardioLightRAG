#!/usr/bin/env python
"""
Generate Document Embeddings for CARDIO-LR

This script encodes medical documents in the dataset using the biomedical sentence transformer
and saves the embeddings for efficient vector-based retrieval. This improves search performance
and allows for accurate document retrieval without requiring FAISS.

Author: CARDIO-LR Team
Date: June 2025
"""

import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

# Add project root to path for imports
sys.path.append('/u/shwkir9t/Documents/Projects/CARDIO-LR')
from config import DATA_PROCESSED_EMBEDDINGS_DIR, BIOBERT_MODEL

def generate_embeddings(batch_size=32, max_docs=None):
    """
    Generate and save document embeddings for the biomedical corpus.
    
    Args:
        batch_size (int): Number of documents to encode at once
        max_docs (int, optional): Maximum number of documents to process, useful for testing
    """
    print("\n=== Generating Document Embeddings ===\n")
    
    # Make sure the embeddings directory exists
    os.makedirs(DATA_PROCESSED_EMBEDDINGS_DIR, exist_ok=True)
    
    # Load documents
    docs_file = os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_documents.json')
    if not os.path.exists(docs_file):
        print(f"Error: Document file {docs_file} not found!")
        return False
    
    with open(docs_file) as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    
    # Limit documents if specified
    if max_docs and max_docs < len(documents):
        print(f"Limiting to {max_docs} documents as specified")
        documents = documents[:max_docs]
    
    # Force CPU usage to avoid CUDA compatibility issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load encoder model
    print(f"Loading sentence transformer model: {BIOBERT_MODEL}")
    model = SentenceTransformer(BIOBERT_MODEL, device="cpu")
    
    # Process documents in batches
    all_embeddings = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"Processing {len(documents)} documents in {total_batches} batches")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
        batch = documents[i:i+batch_size]
        
        # Get text from documents
        if isinstance(batch[0], str):
            texts = batch
        elif isinstance(batch[0], dict) and 'text' in batch[0]:
            texts = [doc['text'] for doc in batch]
        else:
            # Handle other document formats
            texts = [str(doc) for doc in batch]
        
        # Encode batch
        batch_embeddings = model.encode(texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    # Combine all batch results
    embeddings = np.vstack(all_embeddings)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Save embeddings
    output_file = os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'document_embeddings.npy')
    np.save(output_file, embeddings)
    print(f"Saved embeddings to {output_file}")
    
    # Generate metadata about the embeddings
    metadata = {
        'document_count': len(documents),
        'embedding_dim': embeddings.shape[1],
        'model': BIOBERT_MODEL,
        'creation_date': '2025-06-25'
    }
    
    metadata_file = os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'embeddings_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate document embeddings for CARDIO-LR")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to process")
    args = parser.parse_args()
    
    generate_embeddings(batch_size=args.batch_size, max_docs=args.max_docs)