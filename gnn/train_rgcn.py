# gnn/train_rgcn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
from torch.optim import Adam
import sys
import time
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to properly import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED_KG_DIR, DATA_PROCESSED_MODELS_DIR

# Simplified RGCN model that doesn't rely on PyTorch Geometric
class SimpleRGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim=64, hidden_dim=128):
        super(SimpleRGCN, self).__init__()
        # Force CPU usage due to CUDA compatibility mismatch
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Create relation-specific weight matrices
        self.weight1 = nn.Parameter(torch.Tensor(num_relations, embedding_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.Tensor(num_relations, hidden_dim, embedding_dim))
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Model initialized with {embedding_dim} embedding dimensions and {hidden_dim} hidden dimensions")
        
    def forward(self, node_ids, edge_index, edge_type):
        # Get node embeddings
        x = self.embedding(node_ids)
        
        # First layer - more memory efficient implementation
        x1 = self._propagate_chunked(x, edge_index, edge_type, self.weight1, self.hidden_dim)
        x1 = F.relu(self.layer_norm1(x1))
        x1 = self.dropout(x1)
        
        # Second layer
        x2 = self._propagate_chunked(x1, edge_index, edge_type, self.weight2, self.embedding_dim)
        x2 = self.layer_norm2(x2)
        
        # Skip connection
        x_final = x2 + x
        
        return x_final
    
    def _propagate_chunked(self, x, edge_index, edge_type, weight, output_dim, chunk_size=10000):
        """Memory-efficient implementation of message passing using chunking"""
        src, dst = edge_index[0], edge_index[1]
        # Initialize output tensor with correct dimensions
        out = torch.zeros((x.size(0), output_dim), device=self.device)
        
        # Process edges in chunks to reduce memory usage
        num_edges = len(src)
        num_chunks = (num_edges + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_edges)
            
            # Get current chunk of edges
            src_chunk = src[start_idx:end_idx]
            dst_chunk = dst[start_idx:end_idx]
            rel_chunk = edge_type[start_idx:end_idx]
            
            # Group by relation type within chunk
            for rel in range(self.num_relations):
                mask = (rel_chunk == rel)
                if not mask.any():
                    continue
                    
                src_rel = src_chunk[mask]
                dst_rel = dst_chunk[mask]
                
                # Get source embeddings and transform
                src_embedding = x[src_rel]
                transformed = torch.matmul(src_embedding, weight[rel])
                
                # Aggregate at destination nodes
                for i in range(len(dst_rel)):
                    out[dst_rel[i]] += transformed[i]
        
        return out
    
    def get_embeddings(self, node_ids, edge_index, edge_type):
        return self.forward(node_ids, edge_index, edge_type)
    
    def train_model(self, train_data, optimizer, num_epochs=10, batch_size=64, save_path=None):
        """Train the RGCN model with memory-efficient batching"""
        node_ids, edge_index, edge_type, pos_samples, neg_samples = train_data
        
        # Number of batches
        num_samples = len(pos_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        
        logger.info(f"Starting training with {num_samples} samples, {num_batches} batches")
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            # Shuffle training data
            indices = torch.randperm(num_samples)
            pos_samples_shuffled = pos_samples[indices]
            neg_samples_shuffled = neg_samples[indices]
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx in progress_bar:
                try:
                    # Get batch data
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    
                    batch_pos = pos_samples_shuffled[start_idx:end_idx]
                    batch_neg = neg_samples_shuffled[start_idx:end_idx]
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Get embeddings for all nodes - use node subset for memory efficiency
                    node_embeddings = self.forward(node_ids, edge_index, edge_type)
                    
                    # Compute scores for positive and negative samples
                    pos_scores = self._score_triples(node_embeddings, batch_pos)
                    neg_scores = self._score_triples(node_embeddings, batch_neg)
                    
                    # Compute margin loss
                    loss = F.margin_ranking_loss(
                        pos_scores, 
                        neg_scores, 
                        torch.ones(pos_scores.size(0), device=self.device),
                        margin=1.0
                    )
                    
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    # Clean up to reduce memory usage
                    del node_embeddings, pos_scores, neg_scores, loss
                    if batch_idx % 10 == 0:
                        gc.collect()
                        
                except RuntimeError as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    if "out of memory" in str(e).lower():
                        logger.error("Memory error detected. Reducing batch size and continuing...")
                        batch_size = max(batch_size // 2, 1)
                        logger.info(f"New batch size: {batch_size}")
                    else:
                        raise e
                    
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save model if loss improved
            if avg_loss < best_loss and save_path:
                best_loss = avg_loss
                self.save_model(save_path)
                logger.info(f"Model saved to {save_path} (loss: {best_loss:.4f})")
    
    def _score_triples(self, node_embeddings, triples):
        """Compute similarity scores for triples"""
        heads = node_embeddings[triples[:, 0]]
        tails = node_embeddings[triples[:, 2]]
        
        # Simple dot product similarity
        scores = torch.sum(heads * tails, dim=1)
        return scores
    
    def save_model(self, path):
        """Save model state to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load model state from disk"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        logger.info(f"Model loaded from {path}")

def load_kg(kg_path):
    """Load the knowledge graph from disk"""
    logger.info(f"Loading knowledge graph from {kg_path}...")
    start_time = time.time()
    try:
        kg = nx.read_gpickle(kg_path)
        load_time = time.time() - start_time
        logger.info(f"Knowledge graph loaded with {len(kg.nodes)} nodes and {len(kg.edges)} edges in {load_time:.2f} seconds")
        return kg
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {str(e)}")
        raise

def prepare_training_data(kg, batch_size=1000, neg_samples_ratio=1, max_samples=50000):
    """
    Prepare training data for RGCN from the knowledge graph
    Returns full graph in PyTorch format and training samples
    """
    device = torch.device('cpu')
    logger.info(f"Preparing training data on device: {device}")
    
    # Create node mapping for the graph
    logger.info("Creating node mappings...")
    node_mapping = {node: i for i, node in enumerate(kg.nodes())}
    
    # Create relation mapping
    relation_types = set()
    for _, _, data in kg.edges(data=True):
        rel_type = data.get('relation', 'related_to')
        relation_types.add(rel_type)
    
    rel_to_idx = {rel: i for i, rel in enumerate(sorted(relation_types))}
    logger.info(f"Found {len(rel_to_idx)} different relation types")
    
    # Prepare edge index and type arrays
    src_nodes = []
    dst_nodes = []
    edge_types = []
    
    # Process all edges for the graph structure
    logger.info("Converting graph to PyTorch format...")
    edge_list = list(kg.edges(data=True))
    
    # Use a subset of edges if the graph is too large
    if len(edge_list) > max_samples * 2:
        logger.info(f"Graph is very large ({len(edge_list)} edges). Using a random subset of edges.")
        selected_indices = np.random.choice(len(edge_list), max_samples * 2, replace=False)
        edge_list = [edge_list[i] for i in selected_indices]
        logger.info(f"Selected {len(edge_list)} edges for training")
    
    for src, dst, data in tqdm(edge_list, desc="Processing edges"):
        # Convert to indices
        src_idx = node_mapping[src]
        dst_idx = node_mapping[dst]
        
        # Get relation type
        rel_type = data.get('relation', 'related_to')
        rel_idx = rel_to_idx[rel_type]
        
        # Add to arrays
        src_nodes.append(src_idx)
        dst_nodes.append(dst_idx)
        edge_types.append(rel_idx)
    
    # Convert to PyTorch tensors
    logger.info("Converting to PyTorch tensors...")
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)
    edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
    
    # Node IDs tensor
    node_ids = torch.arange(len(node_mapping), device=device)
    
    # Create training samples
    logger.info("Creating training samples...")
    
    # Limit the number of positive samples to avoid memory issues
    max_positive_samples = min(len(edge_list), max_samples)
    logger.info(f"Using {max_positive_samples} positive samples")
    
    # Choose a subset of edges for positive samples
    edge_subset = list(zip(src_nodes, dst_nodes, edge_types))
    selected_edges = np.random.choice(len(edge_subset), max_positive_samples, replace=False)
    
    # Create positive samples from the selected edges
    positive_samples = []
    for idx in tqdm(selected_edges, desc="Generating positive samples"):
        src_idx, dst_idx, rel_idx = edge_subset[idx]
        positive_samples.append([src_idx, rel_idx, dst_idx])
    
    # Create negative samples by corrupting heads or tails
    num_nodes = len(node_mapping)
    negative_samples = []
    
    logger.info("Generating negative samples...")
    num_neg_samples = int(len(positive_samples) * neg_samples_ratio)
    
    for _ in tqdm(range(num_neg_samples), desc="Generating negative samples"):
        # Select a random positive sample
        pos_idx = np.random.randint(0, len(positive_samples))
        h, r, t = positive_samples[pos_idx]
        
        # Corrupt head or tail
        if np.random.random() < 0.5:
            # Corrupt head
            corrupt_h = h
            while corrupt_h == h:
                corrupt_h = np.random.randint(0, num_nodes)
            negative_samples.append([corrupt_h, r, t])
        else:
            # Corrupt tail
            corrupt_t = t
            while corrupt_t == t:
                corrupt_t = np.random.randint(0, num_nodes)
            negative_samples.append([h, r, corrupt_t])
    
    # Convert to PyTorch tensors
    logger.info("Converting samples to tensors...")
    positive_samples = torch.tensor(positive_samples, dtype=torch.long, device=device)
    negative_samples = torch.tensor(negative_samples, dtype=torch.long, device=device)
    
    logger.info(f"Training data prepared: {len(positive_samples)} positive samples, {len(negative_samples)} negative samples")
    
    return node_ids, edge_index, edge_type, positive_samples, negative_samples, len(rel_to_idx)

def create_smaller_graph(kg, max_nodes=10000):
    """Create a smaller subgraph for testing purposes"""
    logger.info(f"Creating smaller graph with max {max_nodes} nodes...")
    
    # Select a random subset of nodes
    all_nodes = list(kg.nodes())
    if len(all_nodes) > max_nodes:
        selected_nodes = np.random.choice(all_nodes, max_nodes, replace=False)
        smaller_kg = kg.subgraph(selected_nodes)
        logger.info(f"Created smaller graph with {len(smaller_kg.nodes())} nodes and {len(smaller_kg.edges())} edges")
        return smaller_kg
    else:
        logger.info(f"Original graph is already small enough: {len(kg.nodes())} nodes")
        return kg

def train_rgcn(kg_path=None, output_path=None, embedding_dim=64, hidden_dim=128, 
               batch_size=64, num_epochs=5, learning_rate=0.001, neg_samples_ratio=1, 
               use_small_graph=False, max_nodes=10000):
    """Train the RGCN model on the knowledge graph"""
    try:
        # Default paths
        if kg_path is None:
            kg_path = os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        
        if output_path is None:
            output_path = os.path.join(DATA_PROCESSED_MODELS_DIR, 'rgcn_model.pt')
        
        # Load knowledge graph
        kg = load_kg(kg_path)
        
        # Create a smaller graph if requested (useful for testing)
        if use_small_graph:
            kg = create_smaller_graph(kg, max_nodes=max_nodes)
        
        # Prepare training data with memory-efficient approach
        node_ids, edge_index, edge_type, pos_samples, neg_samples, num_relations = prepare_training_data(
            kg, batch_size=batch_size, neg_samples_ratio=neg_samples_ratio, max_samples=50000
        )
        
        # Create model
        num_nodes = len(kg.nodes())
        logger.info(f"Creating RGCN model with {num_nodes} nodes and {num_relations} relation types")
        logger.info(f"Embedding dimension: {embedding_dim}, Hidden dimension: {hidden_dim}")
        
        model = SimpleRGCN(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        logger.info(f"Training RGCN model for {num_epochs} epochs")
        train_data = (node_ids, edge_index, edge_type, pos_samples, neg_samples)
        model.train_model(train_data, optimizer, num_epochs=num_epochs, batch_size=batch_size, save_path=output_path)
        
        logger.info(f"RGCN model trained and saved to {output_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGCN model on knowledge graph")
    parser.add_argument("--kg_path", help="Path to knowledge graph pickle file")
    parser.add_argument("--output_path", help="Path to save trained model")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--neg_samples_ratio", type=float, default=1, help="Ratio of negative to positive samples")
    parser.add_argument("--use_small_graph", action="store_true", help="Use a smaller subset of the graph for testing")
    parser.add_argument("--max_nodes", type=int, default=10000, help="Maximum number of nodes when using small graph")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(DATA_PROCESSED_MODELS_DIR, exist_ok=True)
    
    # Train the model with memory optimizations and error handling
    train_rgcn(
        kg_path=args.kg_path,
        output_path=args.output_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        neg_samples_ratio=args.neg_samples_ratio,
        use_small_graph=args.use_small_graph,
        max_nodes=args.max_nodes
    )