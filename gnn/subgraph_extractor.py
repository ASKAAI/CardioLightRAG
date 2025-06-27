# gnn/subgraph_extractor.py
import os
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from config import DATA_PROCESSED_KG_DIR
import pickle

# Import the safe_read_gpickle function from knowledge_integrator
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg_construction.knowledge_integrator import safe_read_gpickle

class SubgraphExtractor:
    def __init__(self, device=None):
        # Set device (GPU if available)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Subgraph extractor using device: {self.device}")
        
        # Load integrated knowledge graph
        print("Loading integrated knowledge graph...")
        kg_path = os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        if not os.path.exists(kg_path):
            print(f"Warning: Knowledge graph file not found at {kg_path}")
            print("Creating an empty graph for testing purposes")
            self.kg = nx.DiGraph()
        else:
            self.kg = safe_read_gpickle(kg_path)
        print(f"Knowledge graph loaded with {len(self.kg.nodes)} nodes and {len(self.kg.edges)} edges")
        
        # Create node and relation mappings
        self._create_mappings()
    
    def _create_mappings(self):
        """Create node and relation type mappings for efficient indexing"""
        print("Creating node and relation mappings...")
        
        # Create node ID mapping
        self.node_to_idx = {node: i for i, node in enumerate(self.kg.nodes())}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        
        # Create relation type mapping
        relation_types = set()
        for _, _, data in self.kg.edges(data=True):
            rel_type = data.get('relation', 'related_to')
            relation_types.add(rel_type)
        
        self.rel_to_idx = {rel: i for i, rel in enumerate(sorted(relation_types))}
        self.idx_to_rel = {i: rel for rel, i in self.rel_to_idx.items()}
        
        print(f"Created mappings for {len(self.node_to_idx)} nodes and {len(self.rel_to_idx)} relation types")
    
    def extract_subgraph(self, entities, hops=2):
        """Extract relevant subgraph based on entities"""
        # Find seed nodes
        # Enhance seed node matching logic to handle variations in entity names
        seed_nodes = []
        for entity in entities:
            for node, data in self.kg.nodes(data=True):
                if 'name' in data and data['name'] is not None:
                    node_name = str(data['name']).strip().lower()
                    entity_name = str(entity['name']).strip().lower()
                    if node_name == entity_name or entity_name in node_name:
                        seed_nodes.append(node)

        if not seed_nodes:
            print(f"Warning: No seed nodes found for entities: {[e['name'] for e in entities]}")
            return nx.DiGraph()  # Return empty graph if no seed nodes found
        
        print(f"Found {len(seed_nodes)} seed nodes for extraction")
        
        # Extract k-hop subgraph
        subgraph_nodes = set(seed_nodes)
        current_frontier = set(seed_nodes)
        
        for hop in range(hops):
            print(f"Extracting hop {hop+1}/{hops}...")
            new_frontier = set()
            
            for node in tqdm(current_frontier, desc=f"Processing frontier nodes at hop {hop+1}"):
                # Check if node exists in the knowledge graph
                if node in self.kg:
                    neighbors = list(self.kg.neighbors(node))
                    new_frontier.update(neighbors)
            
            # Update the frontier and subgraph nodes
            current_frontier = new_frontier - subgraph_nodes
            subgraph_nodes.update(new_frontier)
            print(f"Hop {hop+1}: Added {len(new_frontier)} nodes, total {len(subgraph_nodes)} nodes")
        
        # Extract the subgraph
        subgraph = self.kg.subgraph(subgraph_nodes)
        print(f"Extracted subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
        
        return subgraph
    
    def networkx_to_pytorch(self, graph):
        """
        Convert NetworkX graph to PyTorch Geometric data format for use with RGCN
        
        Returns:
            tuple: (node_ids, edge_index, edge_type, node_mapping)
        """
        # Create node mapping for this specific subgraph
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        reverse_mapping = {i: node for node, i in node_mapping.items()}
        
        # Prepare edge index and type arrays
        edge_index = []
        edge_type = []
        
        # Process all edges
        for src, dst, data in graph.edges(data=True):
            # Convert to local indices
            src_idx = node_mapping[src]
            dst_idx = node_mapping[dst]
            
            # Get relation type
            rel_type = data.get('relation', 'related_to')
            if rel_type in self.rel_to_idx:
                rel_idx = self.rel_to_idx[rel_type]
            else:
                rel_idx = len(self.rel_to_idx) - 1  # Default to last relation type if not found
            
            # Add to arrays
            edge_index.append([src_idx, dst_idx])
            edge_type.append(rel_idx)
        
        # Convert to PyTorch tensors
        if edge_index:  # Check if there are any edges
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)
        else:
            # Create empty tensors if no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_type = torch.zeros((0,), dtype=torch.long, device=self.device)
        
        # Node IDs tensor (just sequential IDs for the subgraph)
        node_ids = torch.arange(len(node_mapping), device=self.device)
        
        return node_ids, edge_index, edge_type, reverse_mapping
    
    def get_subgraph_data(self, entities, hops=2):
        """
        Extract subgraph and convert to PyTorch Geometric format in one step
        
        Returns:
            tuple: (subgraph, node_ids, edge_index, edge_type, node_mapping)
        """
        subgraph = self.extract_subgraph(entities, hops)
        node_ids, edge_index, edge_type, node_mapping = self.networkx_to_pytorch(subgraph)
        
        return subgraph, node_ids, edge_index, edge_type, node_mapping
    
    def create_training_samples(self, graph, neg_samples_ratio=1):
        """
        Create positive and negative training samples from a graph
        
        Args:
            graph: NetworkX graph
            neg_samples_ratio: Ratio of negative samples to positive samples
            
        Returns:
            tuple: (positive_samples, negative_samples)
        """
        # Convert graph to local index mapping
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        
        # Create positive samples
        positive_samples = []
        for src, dst, data in graph.edges(data=True):
            # Convert to local indices
            src_idx = node_mapping[src]
            dst_idx = node_mapping[dst]
            
            # Get relation type
            rel_type = data.get('relation', 'related_to')
            if rel_type in self.rel_to_idx:
                rel_idx = self.rel_to_idx[rel_type]
            else:
                rel_idx = len(self.rel_to_idx) - 1
            
            positive_samples.append([src_idx, rel_idx, dst_idx])
        
        # Create negative samples by corrupting heads or tails
        negative_samples = []
        num_nodes = len(node_mapping)
        
        for _ in range(int(len(positive_samples) * neg_samples_ratio)):
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
        
        return positive_samples, negative_samples
    
    def subgraph_to_text(self, subgraph):
        """Convert subgraph to textual representation"""
        text = "Medical Knowledge Subgraph:\n"
        for node in subgraph.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            text += f"\n- {node_data.get('name', node_id)} ({node_data.get('type', 'Concept')}): "
            
            # Get relations
            relations = []
            for neighbor in subgraph.neighbors(node_id):
                edge_data = subgraph.get_edge_data(node_id, neighbor)
                rel_type = edge_data.get('relation', 'related_to')
                neighbor_data = subgraph.nodes[neighbor]
                neighbor_name = neighbor_data.get('name', neighbor)
                relations.append(f"{rel_type} {neighbor_name}")
            
            text += ", ".join(relations)
        return text

if __name__ == "__main__":
    extractor = SubgraphExtractor()
    entities = [{'name': 'Angina', 'type': 'Disease'}]
    subgraph = extractor.extract_subgraph(entities)
    print("Subgraph extracted with", len(subgraph.nodes), "nodes")
    
    # Convert to PyTorch format
    node_ids, edge_index, edge_type, node_mapping = extractor.networkx_to_pytorch(subgraph)
    print(f"Converted to PyTorch: {len(node_ids)} nodes, {edge_index.shape[1]} edges")
    
    # Create training samples
    pos_samples, neg_samples = extractor.create_training_samples(subgraph)
    print(f"Created {len(pos_samples)} positive and {len(neg_samples)} negative training samples")