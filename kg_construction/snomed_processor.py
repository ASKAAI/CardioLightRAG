import os
import glob
import pandas as pd
import networkx as nx
from tqdm import tqdm
from config import SNOMED_DIR, DATA_PROCESSED_KG_DIR

class SNOMEDProcessor:
    def __init__(self):
        # Automatically detect latest 2025 files
        self.concepts_file = self._get_latest_file("sct2_Concept_Snapshot_*2025*.txt")
        self.descriptions_file = self._get_latest_file("sct2_Description_Snapshot-en_*2025*.txt")
        self.relationships_file = self._get_latest_file("sct2_Relationship_Snapshot_*2025*.txt")
        self.graph = nx.DiGraph()
    
    def _get_latest_file(self, pattern):
        """Get the most recent file matching pattern"""
        files = glob.glob(os.path.join(SNOMED_DIR, pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching: {pattern}")
        return sorted(files, reverse=True)[0]  # Get most recent
    
    def load_data(self):
        print("Loading SNOMED CT 2025 data...")
        # Read concepts with only necessary columns
        self.concepts = pd.read_csv(
            self.concepts_file, 
            sep='\t',
            usecols=['id', 'effectiveTime', 'active', 'moduleId', 'definitionStatusId'],
            dtype={'id': str, 'definitionStatusId': str}
        )
        
        # Filter to active concepts only
        self.concepts = self.concepts[self.concepts['active'] == 1]
        print(f"Loaded {len(self.concepts)} active concepts")
        
        # Read descriptions (English only)
        self.descriptions = pd.read_csv(
            self.descriptions_file, 
            sep='\t',
            usecols=['id', 'conceptId', 'term', 'typeId', 'languageCode', 'active'],
            dtype={'conceptId': str, 'term': str}
        )
        self.descriptions = self.descriptions[
            (self.descriptions['languageCode'] == 'en') & 
            (self.descriptions['active'] == 1)
        ]
        print(f"Loaded {len(self.descriptions)} active English descriptions")
        
        # Read relationships
        self.relationships = pd.read_csv(
            self.relationships_file, 
            sep='\t',
            usecols=['id', 'sourceId', 'destinationId', 'typeId', 'active'],
            dtype={'sourceId': str, 'destinationId': str, 'typeId': str}
        )
        self.relationships = self.relationships[self.relationships['active'] == 1]
        print(f"Loaded {len(self.relationships)} active relationships")
    
    def build_graph(self):
        print("Building SNOMED CT 2025 graph...")
        # Create concept ID to term mapping
        concept_terms = {}
        for _, row in tqdm(self.descriptions.iterrows(), desc="Indexing terms"):
            if row['typeId'] == '900000000000003001':  # Fully Specified Name
                concept_terms[row['conceptId']] = row['term']
        
        print(f"Created term mappings for {len(concept_terms)} concepts")
        
        # Debug: Check if concepts exist in the term mapping
        concept_count = 0
        for _, row in self.concepts.iterrows():
            concept_id = row['id']
            if concept_id in concept_terms:
                concept_count += 1
        
        print(f"Found {concept_count} concepts that have corresponding terms")
        
        # If no concepts have terms, try with different typeId values
        if concept_count == 0:
            print("Trying with all description types instead of just Fully Specified Names...")
            concept_terms = {}
            type_counts = {}
            for _, row in self.descriptions.iterrows():
                type_id = row['typeId']
                if type_id not in type_counts:
                    type_counts[type_id] = 0
                type_counts[type_id] += 1
                
                # Use any description type
                concept_terms[row['conceptId']] = row['term']
            
            print(f"Description type counts: {type_counts}")
            print(f"Created term mappings for {len(concept_terms)} concepts")
        
        # Add concept nodes - use tqdm to show progress
        for _, row in tqdm(self.concepts.iterrows(), total=len(self.concepts), desc="Adding concepts"):
            concept_id = row['id']
            if concept_id in concept_terms:
                self.graph.add_node(
                    concept_id, 
                    name=concept_terms[concept_id],
                    type="Concept",
                    status=row['definitionStatusId']
                )
        
        print(f"Added {len(self.graph.nodes)} nodes to the graph")
        
        # Add relationships
        relationship_count = 0
        for _, row in tqdm(self.relationships.iterrows(), total=len(self.relationships), desc="Adding relationships"):
            source_id = row['sourceId']
            target_id = row['destinationId']
            
            if source_id in self.graph and target_id in self.graph:
                # Get relationship type name
                rel_type = "Is a" if row['typeId'] == '116680003' else "Related to"
                
                self.graph.add_edge(
                    source_id, 
                    target_id, 
                    relation=rel_type,
                    type_id=row['typeId']
                )
                relationship_count += 1
        
        print(f"Added {relationship_count} edges to the graph")
        
        # Save graph
        os.makedirs(DATA_PROCESSED_KG_DIR, exist_ok=True)
        nx.write_gpickle(self.graph, os.path.join(DATA_PROCESSED_KG_DIR, 'snomed_graph_2025.pkl'))
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph

if __name__ == "__main__":
    processor = SNOMEDProcessor()
    processor.load_data()
    processor.build_graph()