import re
from transformers import pipeline
from config import DATA_PROCESSED_KG_DIR
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg_construction.knowledge_integrator import safe_read_gpickle

class PatientContextProcessor:
    def __init__(self):
        self.ner = pipeline(
            "token-classification", 
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        self.umls_graph = safe_read_gpickle(
            os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        )
        self.entity_types = {
            'Condition', 'Medication', 'Procedure', 
            'LabResult', 'Allergy', 'Demographic'
        }
    
    def extract_entities(self, context_text):
        """Extract medical entities from patient context"""
        entities = self.ner(context_text)
        return [
            {
                'text': ent['word'], 
                'type': ent['entity_group'],
                'start': ent['start'],
                'end': ent['end']
            }
            for ent in entities if ent['entity_group'] in self.entity_types
        ]
    
    def link_to_knowledge(self, entities):
        """Link entities to knowledge graph concepts"""
        linked_entities = []
        
        for entity in entities:
            best_match = None
            best_score = 0
            
            # Search for matching concepts in knowledge graph
            for node, data in self.umls_graph.nodes(data=True):
                if 'name' in data:
                    # Simple string matching (could be enhanced with embeddings)
                    score = self.similarity_score(entity['text'], data['name'])
                    if score > best_score:
                        best_match = {
                            'cui': node,
                            'name': data['name'],
                            'type': data.get('type', 'Concept')
                        }
                        best_score = score
            
            if best_match and best_score > 0.6:  # Minimum similarity threshold
                linked_entities.append({
                    'patient_entity': entity,
                    'kg_entity': best_match
                })
        
        return linked_entities
    
    def similarity_score(self, text1, text2):
        """Simple similarity score between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))
    
    def parse_context(self, context_text):
        """Full pipeline: extract and link entities"""
        entities = self.extract_entities(context_text)
        linked_entities = self.link_to_knowledge(entities)
        return linked_entities