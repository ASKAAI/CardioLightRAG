from gnn.subgraph_extractor import SubgraphExtractor

class ContextIntegrator:
    def __init__(self):
        self.subgraph_extractor = SubgraphExtractor()
    
    def integrate_patient_context(self, query_entities, patient_entities):
        """Integrate patient context into the query processing"""
        # Combine query and patient entities
        all_entities = query_entities.copy()
        
        # Add patient entities to the list
        for entity in patient_entities:
            if isinstance(entity, dict) and 'kg_entity' in entity:
                kg_entity = entity['kg_entity']
                all_entities.append({
                    'id': kg_entity['cui'],
                    'name': kg_entity['name'],
                    'type': kg_entity['type']
                })
        
        # Extract personalized subgraph
        personalized_subgraph = self.subgraph_extractor.extract_subgraph(all_entities)
        
        # Add patient-specific relationships
        self.add_patient_relationships(personalized_subgraph, patient_entities)
        
        return personalized_subgraph
    
    def add_patient_relationships(self, subgraph, patient_entities):
        """Add patient-specific relationships to the subgraph"""
        # This would be enhanced with medical logic in a real system
        # For demo: add "has_condition" and "taking_medication" relationships

        patient_node = "PATIENT_001"  # Represent the patient

        # Add patient node
        subgraph.add_node(patient_node, name="Current Patient", type="Patient")

        # Add relationships to conditions and medications
        for entity in patient_entities:
            if isinstance(entity, dict) and 'kg_entity' in entity:
                kg_entity = entity['kg_entity']
                if kg_entity['type'] == 'Condition':
                    subgraph.add_edge(
                        patient_node, 
                        kg_entity['cui'], 
                        relation="has_condition",
                        source="patient_context"
                    )
                elif kg_entity['type'] == 'Medication':
                    subgraph.add_edge(
                        patient_node, 
                        kg_entity['cui'], 
                        relation="taking_medication",
                        source="patient_context"
                    )
            else:
                print(f"Skipping invalid entity: {entity}")

        return subgraph
    
    def format_context_insights(self, context_insights):
        """Format patient context insights into a structured text format"""
        formatted_context = []

        if 'description' in context_insights:
            formatted_context.append(f"Description: {context_insights['description']}")

        if 'conditions' in context_insights and context_insights['conditions']:
            formatted_context.append("Conditions:")
            formatted_context.extend([f"- {condition}" for condition in context_insights['conditions']])

        if 'medications' in context_insights and context_insights['medications']:
            formatted_context.append("Medications:")
            formatted_context.extend([f"- {medication}" for medication in context_insights['medications']])

        if 'allergies' in context_insights and context_insights['allergies']:
            formatted_context.append("Allergies:")
            formatted_context.extend([f"- {allergy}" for allergy in context_insights['allergies']])

        return "\n".join(formatted_context)