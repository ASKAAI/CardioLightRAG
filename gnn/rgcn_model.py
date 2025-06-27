# gnn/rgcn_model.py
import os
import numpy as np

# Create a mock RGCN class that can be imported without dependencies
# This allows pipeline.py to import RGCN without needing torch_geometric
class RGCN:
    """Mock RGCN class for import compatibility"""
    def __init__(self, num_relations=0, num_bases=None, hidden_channels=0, embedding_dim=128, hidden_dim=256, dropout=0.2):
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def to(self, device):
        return self
        
    def load_state_dict(self, state_dict, map_location=None):
        # Mock implementation
        return self
        
    def forward(self, node_ids, edge_index, edge_type):
        # Mock forward pass that returns random embeddings of the right shape
        try:
            import torch
            device = getattr(self, 'device', 'cpu')
            num_nodes = len(node_ids)
            # Return random embeddings with the same shape as expected
            return torch.randn(num_nodes, self.embedding_dim, device=device)
        except ImportError:
            # If torch isn't available, use numpy instead
            num_nodes = len(node_ids)
            return np.random.randn(num_nodes, self.embedding_dim)
    
    def __call__(self, *args, **kwargs):
        # Make the mock object callable to simulate the PyTorch module behavior
        return self.forward(*args, **kwargs)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import RGCNConv
    import os
    from tqdm import tqdm
    import numpy as np

    class RGCNReasoner(nn.Module):
        def __init__(self, num_nodes, num_relations, embedding_dim=128, hidden_dim=256, num_bases=None, dropout=0.2, **kwargs):
            super().__init__()
            # Support hidden_channels as an alias for hidden_dim
            if 'hidden_channels' in kwargs:
                hidden_dim = kwargs['hidden_channels']
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.embedding = nn.Embedding(num_nodes, embedding_dim)
            
            # Calculate number of bases if not provided (usually 30% of num_relations is a good default)
            if num_bases is None:
                num_bases = max(1, int(0.3 * num_relations))
                
            # Use bases decomposition for better performance with many relation types
            self.conv1 = RGCNConv(embedding_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.conv3 = RGCNConv(hidden_dim, embedding_dim, num_relations, num_bases=num_bases)
            
            self.dropout = nn.Dropout(dropout)
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            
            # Move model to GPU if available
            self.to(self.device)
            
        def forward(self, node_ids, edge_index, edge_type):
            # Move inputs to the same device as the model
            if not isinstance(node_ids, torch.Tensor):
                node_ids = torch.tensor(node_ids, device=self.device)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, device=self.device)
            if not isinstance(edge_type, torch.Tensor):
                edge_type = torch.tensor(edge_type, device=self.device)
                
            # Forward pass with skip connections and normalization
            x = self.embedding(node_ids)
            x1 = self.conv1(x, edge_index, edge_type)
            x1 = F.relu(self.layer_norm1(x1))
            x1 = self.dropout(x1)
            
            x2 = self.conv2(x1, edge_index, edge_type)
            x2 = F.relu(self.layer_norm2(x2))
            x2 = self.dropout(x2)
            
            # Skip connection with original embeddings
            x3 = self.conv3(x2, edge_index, edge_type)
            x_final = x3 + x  # Residual connection
            
            return x_final
        
        def get_embeddings(self, node_ids, edge_index, edge_type):
            return self.forward(node_ids, edge_index, edge_type)
        
        def train_model(self, train_data, optimizer, num_epochs=10, batch_size=128, save_path=None):
            """
            Train the RGCN model with GPU acceleration
            
            Args:
                train_data: tuple of (node_ids, edge_index, edge_type, positive_samples, negative_samples)
                optimizer: PyTorch optimizer
                num_epochs: Number of training epochs
                batch_size: Batch size for training
                save_path: Path to save the model
            """
            node_ids, edge_index, edge_type, pos_samples, neg_samples = train_data
            
            # Move all training data to device
            node_ids = torch.tensor(node_ids, device=self.device)
            edge_index = torch.tensor(edge_index, device=self.device)
            edge_type = torch.tensor(edge_type, device=self.device)
            pos_samples = torch.tensor(pos_samples, device=self.device)
            neg_samples = torch.tensor(neg_samples, device=self.device)
            
            # Number of batches
            num_samples = len(pos_samples)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            best_loss = float('inf')
            
            for epoch in range(num_epochs):
                self.train()
                total_loss = 0
                
                # Shuffle training data
                indices = torch.randperm(num_samples)
                pos_samples_shuffled = pos_samples[indices]
                neg_samples_shuffled = neg_samples[indices]
                
                progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
                
                for batch_idx in progress_bar:
                    # Get batch data
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    
                    batch_pos = pos_samples_shuffled[start_idx:end_idx]
                    batch_neg = neg_samples_shuffled[start_idx:end_idx]
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Get embeddings for all nodes
                    node_embeddings = self.forward(node_ids, edge_index, edge_type)
                    
                    # Compute scores for positive and negative samples
                    pos_scores = self.score_triples(node_embeddings, batch_pos)
                    neg_scores = self.score_triples(node_embeddings, batch_neg)
                    
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
                    
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
                # Save model if loss improved
                if avg_loss < best_loss and save_path:
                    best_loss = avg_loss
                    self.save_model(save_path)
                    print(f"Model saved to {save_path} (loss: {best_loss:.4f})")
        
        def score_triples(self, node_embeddings, triples):
            """
            Compute similarity scores for (head, relation, tail) triples
            
            Args:
                node_embeddings: Tensor of node embeddings
                triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples
            
            Returns:
                Tensor of scores for each triple
            """
            # Extract embeddings for heads and tails
            heads = node_embeddings[triples[:, 0]]
            tails = node_embeddings[triples[:, 2]]
            
            # Compute simple dot product similarity
            scores = torch.sum(heads * tails, dim=1)
            
            return scores
        
        def save_model(self, path):
            """Save model state to disk"""
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
            
        def load_model(self, path):
            """Load model state from disk"""
            self.load_state_dict(torch.load(path, map_location=self.device))
            self.eval()

    # Update RGCN to the real implementation if dependencies are available
    RGCN = RGCNReasoner
    
except ImportError:
    # Keep the mock RGCN if dependencies are not available
    print("Warning: Full RGCN implementation not available. Using mock implementation.")