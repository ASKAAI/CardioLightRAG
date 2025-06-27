import os
import torch
from retrieval.hybrid_retriever import HybridRetriever
from kg_construction.knowledge_integrator import KnowledgeIntegrator
from gnn.subgraph_extractor import SubgraphExtractor
from gnn.rgcn_model import RGCN
from generation.biomed_generator import BiomedGenerator
from generation.answer_validator import AnswerValidator
from generation.explainability import TraceabilityLogger
from personalization.context_integrator import ContextIntegrator

class CardiologyLightRAG:
    def __init__(self):
        print("Initializing Cardiology LightRAG system...")
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize all pipeline components
        print("Loading retrieval system...")
        self.retriever = HybridRetriever()
        
        print("Loading knowledge graph components...")
        self.knowledge_integrator = KnowledgeIntegrator()
        self.subgraph_extractor = SubgraphExtractor(device=self.device)
        
        print("Loading GNN model for path selection...")
        # Initialize RGCN model with appropriate parameters
        num_relations = len(self.subgraph_extractor.rel_to_idx)
        num_nodes = self.subgraph_extractor.kg.number_of_nodes()
        self.gnn_model = RGCN(
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_bases=4,  # Reduce parameter size with basis decomposition
            hidden_channels=64
        ).to(self.device)
        # Load pretrained weights if available
        self._load_gnn_model()
        
        print("Loading generator and validation components...")
        self.generator = BiomedGenerator()
        # Ensure attention_mask and pad_token_id are set for reliable generation
        self.generator.set_generation_parameters(attention_mask=True, pad_token_id=50256)
        self.validator = AnswerValidator()
        self.trace_logger = TraceabilityLogger()
        self.context_integrator = ContextIntegrator()
        
        print("Cardiology LightRAG system initialized and ready!")

    def _load_gnn_model(self):
        """Load pre-trained GNN model weights if available"""
        model_path = os.path.join('models', 'rgcn_model.pt')
        if os.path.exists(model_path):
            try:
                self.gnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Loaded pre-trained GNN model weights")
            except Exception as e:
                print(f"Could not load pre-trained GNN model: {e}")
                print("Using untrained GNN model")
        else:
            print("No pre-trained GNN model found. Using untrained model.")

    def process_query(self, query, patient_context=None):
        """
        Process a clinical query through the full pipeline
        
        Args:
            query (str): The clinical query
            patient_context (dict, optional): Patient context information
            
        Returns:
            tuple: (answer, explanation)
        """
        # STAGE 1: Query Input & Hybrid Retrieval (BioASQ/MedQuAD)
        print("STAGE 1: Performing hybrid retrieval...")
        vector_results, symbolic_results = self.retriever.hybrid_retrieve(query)
        self.trace_logger.log_retrieval(query, vector_results, symbolic_results)
        
        # Extract medical entities from query and results for KG lookup
        print("Extracting medical entities...")
        medical_entities = self.knowledge_integrator.extract_entities(query, vector_results)
        
        # STAGE 2: Knowledge Graph & Subgraph Generation (DrugBank/UMLS/SNOMED)
        print("STAGE 2: Generating knowledge subgraph...")
        subgraph, node_ids, edge_index, edge_type, node_mapping = self.subgraph_extractor.get_subgraph_data(
            medical_entities, hops=2
        )
        
        # STAGE 3: GNN Path Selection
        print("STAGE 3: Running GNN path selection...")
        # Only run GNN path selection if we have a valid subgraph
        if edge_index.shape[1] > 0:
            # Generate node embeddings
            with torch.no_grad():
                node_embeddings = self.gnn_model(node_ids, edge_index, edge_type)
                
            # Select top paths using node embeddings
            selected_paths = self._select_top_paths(subgraph, node_embeddings, node_mapping)
            subgraph_text = self._format_selected_paths(selected_paths)
            
            # Log the knowledge graph processing
            self.trace_logger.log_knowledge_graph(subgraph, selected_paths)
        else:
            subgraph_text = "No relevant knowledge graph information found."
            self.trace_logger.log_message("No subgraph generated - insufficient entity matches")
        
        # Integrate patient context if available
        if patient_context:
            print("Integrating patient context...")
            context_insights = self.context_integrator.integrate_patient_context(
                medical_entities, patient_context
            )
            context_text = self.context_integrator.format_context_insights(context_insights)
        else:
            context_text = ""
        
        # Create combined context for generation
        context = f"## Retrieved Medical Knowledge\n"
        context += f"**Relevant Documents:**\n"
        for i, doc in enumerate(vector_results[:3]):
            context += f"{i+1}. {doc[:200]}...\n\n"
        
        context += f"\n## Knowledge Graph Information\n{subgraph_text}\n"
        
        if context_text:
            context += f"\n## Patient Context\n{context_text}\n"
        
        # STAGE 4: LLM Response Generation (BioGPT or T5)
        print("STAGE 4: Generating clinical answer...")
        answer = self.generator.generate_answer(context, query)
        
        # STAGE 5: Answer Validation
        print("STAGE 5: Validating clinical accuracy...")
        is_valid, validation_msg = self.validator.validate_answer(context, answer)
        if not is_valid:
            answer = f"{answer}\n\n*Validation Note: {validation_msg}*"
        
        self.trace_logger.log_generation(context, answer)
        
        # Generate explanation of the pipeline process
        explanation = self.trace_logger.generate_explanation()
        
        return answer, explanation
    
    def _select_top_paths(self, subgraph, node_embeddings, node_mapping):
        """
        Select important paths in the knowledge graph using node embeddings
        
        Args:
            subgraph: The NetworkX subgraph
            node_embeddings: PyTorch tensor of node embeddings from GNN
            node_mapping: Mapping from subgraph indices to original KG indices
            
        Returns:
            list: Selected paths as lists of nodes
        """
        # Convert embeddings to numpy for processing
        embeddings_np = node_embeddings.cpu().numpy()
        
        # Identify important nodes based on embedding norms (higher = more important)
        node_importance = {
            i: float(np.linalg.norm(embeddings_np[i])) 
            for i in range(len(node_embeddings))
        }
        
        # Sort nodes by importance
        sorted_nodes = sorted(node_importance.keys(), key=lambda x: node_importance[x], reverse=True)
        top_nodes = sorted_nodes[:min(5, len(sorted_nodes))]
        
        # Find paths between important nodes
        selected_paths = []
        for i, src in enumerate(top_nodes):
            src_node = node_mapping[src]
            for dst in top_nodes[i+1:]:
                dst_node = node_mapping[dst]
                try:
                    # Find shortest path between these important nodes
                    path = nx.shortest_path(subgraph, source=src_node, target=dst_node)
                    if len(path) > 1:  # Only add meaningful paths (more than one node)
                        selected_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # If we couldn't find paths between top nodes, extract paths from/to top nodes
        if not selected_paths and len(top_nodes) > 0:
            for node_idx in top_nodes:
                node = node_mapping[node_idx]
                # Get direct neighbors
                neighbors = list(subgraph.neighbors(node))
                for neighbor in neighbors[:min(3, len(neighbors))]:
                    selected_paths.append([node, neighbor])
        
        return selected_paths[:5]  # Return at most 5 paths
    
    def _format_selected_paths(self, paths):
        """Format selected paths as readable text"""
        if not paths:
            return "No significant knowledge paths identified."
            
        text = "**Key Medical Knowledge Pathways:**\n"
        for i, path in enumerate(paths):
            path_str = " → ".join([self._format_node(node) for node in path])
            text += f"{i+1}. {path_str}\n"
        
        return text
    
    def _format_node(self, node_id):
        """Format a node as readable text"""
        # Try to get node data from the KG
        try:
            node_data = self.subgraph_extractor.kg.nodes[node_id]
            name = node_data.get('name', str(node_id))
            type_str = node_data.get('type', '')
            if type_str:
                return f"{name} ({type_str})"
            return name
        except:
            return str(node_id)

# Add necessary import
import numpy as np
import networkx as nx

# Example usage
if __name__ == "__main__":
    system = CardiologyLightRAG()
    
    # Example cardiology query
    query = "What are the first-line treatments for stable angina in diabetic patients?"
    patient_context = {
        "age": 65,
        "gender": "male",
        "conditions": ["diabetes type 2", "hypertension"],
        "medications": ["metformin", "lisinopril"],
        "allergies": ["aspirin"]
    }
    
    print(f"\nQuestion: {query}")
    if patient_context:
        print(f"Patient Context: {patient_context}")
    
    answer, explanation = system.process_query(query, patient_context)
    
    print("\nClinical Answer:")
    print(answer)
    
    print("\nExplanation:")
    print(explanation)