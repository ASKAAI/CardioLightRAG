# kg_construction/umls_processor.py
import os
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from config import (
    MRCONSO_PATH, MRREL_PATH, MRSTY_PATH, 
    DATA_PROCESSED_KG_DIR, CARDIOLOGY_SEMANTIC_TYPES
)
import pickle

class UMLSProcessor:
    def __init__(self):
        self.concept_df = None
        self.relations_df = None
        self.semantic_types = None
        self.graph = nx.DiGraph()
        
    def load_data(self):
        """Load UMLS RRF files with efficient chunking"""
        print("Loading UMLS concepts...")
        # Load concepts with only necessary columns
        self.concept_df = pd.read_csv(
            MRCONSO_PATH, 
            sep='|', 
            header=None,
            usecols=[0, 1, 2, 3, 4, 6, 7, 11, 12, 14],
            names=['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SAB', 'TTY', 'STR', 'SUPPRESS', 'CVF'],
            dtype={'CUI': 'category', 'LAT': 'category', 'STR': str},
            low_memory=False
        )
        
        # Filter to English concepts
        self.concept_df = self.concept_df[self.concept_df['LAT'] == 'ENG']
        
        print("Loading UMLS relations...")
        # Load relations with chunking for memory efficiency
        chunks = []
        for chunk in tqdm(pd.read_csv(
            MRREL_PATH, 
            sep='|', 
            header=None,
            usecols=[0, 1, 2, 3, 4, 5, 7],
            names=['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2'],
            dtype={'CUI1': 'category', 'CUI2': 'category', 'REL': 'category'},
            chunksize=1000000
        )):
            chunks.append(chunk)
        self.relations_df = pd.concat(chunks)
        
        print("Loading semantic types...")
        self.semantic_types = pd.read_csv(
            MRSTY_PATH, 
            sep='|', 
            header=None,
            usecols=[0, 1, 3],
            names=['CUI', 'TUI', 'STN'],
            dtype={'CUI': 'category', 'TUI': 'category'}
        )
        
        # Filter to cardiology-related semantic types
        self.semantic_types = self.semantic_types[
            self.semantic_types['TUI'].isin(CARDIOLOGY_SEMANTIC_TYPES)
        ]
        
    def build_cardio_graph(self):
        """Build cardiovascular knowledge graph"""
        print("Building cardiology knowledge graph...")
        
        # Filter concepts to cardiology semantic types
        cardio_cuis = set(self.semantic_types['CUI'])
        self.concept_df = self.concept_df[self.concept_df['CUI'].isin(cardio_cuis)]
        
        # Get preferred names for each CUI
        pref_terms = self.concept_df[
            (self.concept_df['TS'] == 'P') & 
            (self.concept_df['STT'] == 'PF') &
            (self.concept_df['SUPPRESS'] == 'N')
        ].groupby('CUI').first()['STR']
        
        # Add nodes to graph
        for cui, name in tqdm(pref_terms.items(), desc="Adding nodes"):
            self.graph.add_node(cui, name=name)
        
        # Filter relations to cardiology concepts
        cardio_relations = self.relations_df[
            self.relations_df['CUI1'].isin(cardio_cuis) & 
            self.relations_df['CUI2'].isin(cardio_cuis)
        ]
        
        # Add relations to graph
        for _, row in tqdm(cardio_relations.iterrows(), total=len(cardio_relations), desc="Adding edges"):
            self.graph.add_edge(row['CUI1'], row['CUI2'], relation=row['REL'])
        
        # Save graph
        os.makedirs(DATA_PROCESSED_KG_DIR, exist_ok=True)
        with open(os.path.join(DATA_PROCESSED_KG_DIR, 'umls_cardio_graph.pkl'), 'wb') as f:
            pickle.dump(self.graph, f, protocol=4)
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph

if __name__ == "__main__":
    processor = UMLSProcessor()
    processor.load_data()
    processor.build_cardio_graph()