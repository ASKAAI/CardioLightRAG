# kg_construction/drugbank_processor.py
import os
import sys
import pandas as pd
import networkx as nx
from config import DRUGBANK_PATH, DRUGBANK_VOCAB_PATH, DATA_PROCESSED_KG_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def integrate_drugbank():
    """Integrate DrugBank data into the knowledge graph"""
    print("Processing DrugBank data...")
    
    # Load DrugBank data
    drugs = pd.read_csv(DRUGBANK_PATH)
    vocab = pd.read_csv(DRUGBANK_VOCAB_PATH)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add drug nodes
    for _, row in drugs.iterrows():
        drug_id = row['drugbank-id']
        G.add_node(
            f"DRUGBANK:{drug_id}",
            name=row['name'],
            description=row['description'],
            type='Drug',
            source='DrugBank'
        )
    
    # Add interactions
    for _, row in vocab.iterrows():
        if 'cas_number_vocab' in row and row['cas_number_vocab'] == 'drug-interactions':
            source = f"DRUGBANK:{row['drugbank_id']}"
            target = f"DRUGBANK:{row['interacting_drug_id']}"
            G.add_edge(source, target, relation='interacts_with')
    
    # Load UMLS graph and integrate
    umls_graph = nx.read_gpickle(os.path.join(DATA_PROCESSED_KG_DIR, 'umls_cardio_graph.pkl'))
    combined_graph = nx.compose(umls_graph, G)
    
    # Add mapping between DrugBank and UMLS
    # This would normally require a mapping table, but we'll create a simple mock
    for node in combined_graph.nodes:
        if node.startswith("DRUGBANK:"):
            drug_name = combined_graph.nodes[node]['name']
            # Find matching UMLS concepts
            for umls_node, data in umls_graph.nodes(data=True):
                if 'name' in data and drug_name.lower() in data['name'].lower():
                    combined_graph.add_edge(node, umls_node, relation='same_as')
                    combined_graph.add_edge(umls_node, node, relation='same_as')
    
    # Save integrated graph
    nx.write_gpickle(combined_graph, os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl'), protocol=4)
    print(f"Integrated graph saved with {len(combined_graph.nodes)} nodes")
    return combined_graph

if __name__ == "__main__":
    integrate_drugbank()