# retrieval/hybrid_retriever.py
"""
Hybrid Retrieval Module for CARDIO-LR

This module implements a hybrid retrieval system that combines vector-based similarity search
with symbolic knowledge graph querying for improved cardiology information retrieval.

The hybrid approach allows the system to benefit from both:
1. Dense vector retrieval: Finding semantically similar medical text
2. Symbolic retrieval: Leveraging structured knowledge about medical concepts and relationships

Author: CARDIO-LR Team
Date: June 2025
"""

import os
import json
import faiss
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from config import DATA_PROCESSED_EMBEDDINGS_DIR, DATA_PROCESSED_KG_DIR, BIOBERT_MODEL
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg_construction.knowledge_integrator import safe_read_gpickle

class HybridRetriever:
    """
    Hybrid retrieval system combining vector-based and knowledge graph-based approaches.
    
    This class implements a dual retrieval strategy for medical information retrieval
    specifically optimized for cardiology queries. It uses FAISS for efficient vector
    similarity search and a knowledge graph for concept-based retrieval.
    
    Attributes:
        index (faiss.Index): FAISS index for vector similarity search
        documents (list): List of medical documents corresponding to index vectors
        kg (networkx.Graph): Knowledge graph of medical concepts and relationships
        model (SentenceTransformer): Biomedical sentence encoder model
    """
    
    def __init__(self):
        """
        Initialize the hybrid retriever with vector index and knowledge graph.
        
        Loads the pre-built FAISS index, document collection, and integrated
        medical knowledge graph focused on cardiology.
        
        Raises:
            FileNotFoundError: If required data files are missing
            RuntimeError: If models fail to load
        """
        # Load FAISS index and documents
        self.index = faiss.read_index(os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_index.faiss'))
        with open(os.path.join(DATA_PROCESSED_EMBEDDINGS_DIR, 'bioasq_documents.json')) as f:
            self.documents = json.load(f)
        
        # Load integrated knowledge graph
        self.kg = safe_read_gpickle(os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl'))
        
        # Initialize encoder
        self.model = SentenceTransformer(BIOBERT_MODEL)
    
    def vector_retrieve(self, query, k=5):
        """
        Retrieve documents using vector similarity search.
        
        Encodes the query and performs approximate nearest neighbor search
        using the FAISS index to find semantically similar documents.
        
        Args:
            query (str): The clinical question or query text
            k (int, optional): Number of results to retrieve. Defaults to 5.
            
        Returns:
            list: The top-k documents most similar to the query
            
        Examples:
            >>> retriever = HybridRetriever()
            >>> results = retriever.vector_retrieve("What causes stable angina?")
        """
        query_embed = self.model.encode([query])
        D, I = self.index.search(query_embed.astype(np.float32), k)
        return [self.documents[i] for i in I[0]]
    
    def symbolic_retrieve(self, query):
        """
        Retrieve relevant entities from knowledge graph based on query.
        
        Searches the knowledge graph for entities mentioned in the query
        and retrieves them along with their associated metadata.
        
        Args:
            query (str): The clinical question or query text
            
        Returns:
            list: List of dictionaries with entity information
            
        Examples:
            >>> retriever = HybridRetriever()
            >>> entities = retriever.symbolic_retrieve("What is the role of beta-blockers in angina?")
        """
        query_lower = query.lower()
        entities = []
        
        # Search in knowledge graph
        for node, data in self.kg.nodes(data=True):
            if 'name' in data and data['name'] is not None and data['name'].lower() in query_lower:
                entities.append({
                    'id': node,
                    'name': data['name'],
                    'type': data.get('type', 'Concept'),
                    'source': data.get('source', 'UMLS')
                })
        return entities
    
    def hybrid_retrieve(self, query, k=5):
        """
        Combine vector and symbolic retrieval for comprehensive results.
        
        Executes both retrieval methods in parallel and returns their results.
        Integration of these results is handled by downstream components.
        
        Args:
            query (str): The clinical question or query text
            k (int, optional): Number of vector results to retrieve. Defaults to 5.
            
        Returns:
            tuple: (vector_results, symbolic_results) containing both types of results
            
        Examples:
            >>> retriever = HybridRetriever()
            >>> vector_results, symbolic_results = retriever.hybrid_retrieve("What are the treatments for angina?")
        """
        vector_results = self.vector_retrieve(query, k)
        symbolic_results = self.symbolic_retrieve(query)
        return vector_results, symbolic_results


if __name__ == "__main__":
    # Example usage demonstration
    retriever = HybridRetriever()
    query = "What are the treatments for angina?"
    vector, symbolic = retriever.hybrid_retrieve(query)
    print("Vector results:", vector[:1])
    print("Symbolic results:", symbolic)