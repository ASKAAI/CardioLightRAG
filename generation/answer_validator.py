from transformers import pipeline
import torch
import re
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

class AnswerValidator:
    def __init__(self):
        # Force CPU usage to avoid CUDA compatibility issues
        self.contradiction_detector = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=-1  # Force CPU usage
        )
        self.medical_fact_checker = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=-1  # Force CPU usage
        )
        
        # Load cardiology-specific terms for domain validation
        self.cardiology_terms = self._load_cardiology_terms()
        
        # Define hallucination detection rules
        self.hallucination_indicators = [
            r"(?i)100\s*%\s*effective",
            r"(?i)cure[sd]?\s+all",
            r"(?i)guaranteed to",
            r"(?i)miracle treatment",
            r"(?i)always works",
            r"(?i)never fails",
            r"(?i)universally effective",
            r"(?i)absolutely no side effects",
            r"(?i)completely safe for all patients",
        ]
        
        # Define contradiction rules based on medical knowledge
        self.contradiction_rules = [
            # Format: (pattern_a, pattern_b) - if both appear, likely contradiction
            (r"(?i)beta.?blockers? .{0,30}contraindicated", r"(?i)beta.?blockers? .{0,30}recommended"),
            (r"(?i)aspirin .{0,30}safe", r"(?i)aspirin .{0,30}allerg"),
            (r"(?i)increase[sd]? blood pressure", r"(?i)decrease[sd]? blood pressure"),
            (r"(?i)ACE inhibitors? .{0,30}safe .{0,30}pregnancy", r"(?i)ACE inhibitors? .{0,50}avoid .{0,30}pregnancy"),
        ]
    
    def _load_cardiology_terms(self) -> List[str]:
        """Load cardiology-specific terminology for domain validation"""
        # In a production system, these would be loaded from a comprehensive file
        return [
            "heart", "cardiac", "cardiovascular", "coronary", "angina", 
            "myocardial infarction", "arrhythmia", "atrial fibrillation", 
            "tachycardia", "bradycardia", "hypertension", "hypotension",
            "stroke", "thrombosis", "stent", "angioplasty", "beta blocker",
            "ace inhibitor", "echocardiogram", "warfarin", "cardiologist",
            "electrocardiogram", "ecg", "ekg", "valve", "pacemaker",
            "defibrillator", "anticoagulant", "cardiomyopathy"
        ]
    
    def check_contradiction(self, context: str, answer: str) -> Tuple[bool, str]:
        """Check if answer contradicts the context using multiple methods"""
        # Method 1: NLI-based contradiction detection
        result = self.contradiction_detector(
            f"{context} [SEP] {answer}",
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        
        if result['labels'][0] == 'contradiction' and result['scores'][0] > 0.7:
            return True, "Model identified logical contradiction between answer and context"
        
        # Method 2: Rule-based contradiction detection
        for pattern_a, pattern_b in self.contradiction_rules:
            if re.search(pattern_a, context) and re.search(pattern_b, answer):
                return True, f"Rule-based contradiction detected: {re.search(pattern_a, context).group(0)} vs {re.search(pattern_b, answer).group(0)}"
            if re.search(pattern_b, context) and re.search(pattern_a, answer):
                return True, f"Rule-based contradiction detected: {re.search(pattern_b, context).group(0)} vs {re.search(pattern_a, answer).group(0)}"
        
        # Method 3: Entity-relation mismatch (check if relations in KG match output)
        # This would require access to the knowledge graph triples
        # Simplified check for proof of concept
        entities_context = self._extract_entities(context)
        entities_answer = self._extract_entities(answer)
        
        for entity in entities_answer:
            if entity not in entities_context:
                # Only flag if it's a significant medical entity, not common words
                if len(entity) > 5 and any(term in entity.lower() for term in self.cardiology_terms):
                    return True, f"Potential hallucination: Entity '{entity}' not found in source context"
        
        return False, ""
    
    def detect_hallucination(self, context: str, answer: str) -> Tuple[bool, str]:
        """Detect hallucinated content in the generated answer"""
        # Method 1: Pattern-based hallucination detection
        for pattern in self.hallucination_indicators:
            match = re.search(pattern, answer)
            if match:
                return True, f"Potential hallucination detected: '{match.group(0)}'"
        
        # Method 2: Check for unsupported specific values/percentages
        percentage_matches = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", answer)
        for percentage in percentage_matches:
            # Look for the same percentage in context
            if percentage not in context:
                return True, f"Unsupported specific value: {percentage}% not found in source"
        
        # Method 3: Check for medical claims without evidence
        statements = self._extract_medical_claims(answer)
        for statement in statements:
            if not self._has_supporting_evidence(statement, context):
                return True, f"Medical claim without supporting evidence: '{statement}'"
        
        return False, ""
    
    def check_incoherence(self, answer: str) -> Tuple[bool, str]:
        """Check if the answer is logically coherent"""
        # Check for self-contradictory statements
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if len(sentences) <= 1:
            return False, ""
        
        # Check for repeated content (a sign of incoherence)
        sentence_similarity = self._compute_sentence_similarity(sentences)
        if sentence_similarity > 0.8:  # High similarity threshold
            return True, "Detected repetitive content suggesting incoherence"
        
        # Check for logical flow disruption
        transitions = ["however", "but", "although", "conversely", "on the other hand"]
        contradictions = 0
        for transition in transitions:
            contradictions += answer.lower().count(transition)
        
        if contradictions >= 3 and len(sentences) < 6:
            return True, "Multiple conflicting transitions suggest incoherent reasoning"
        
        return False, ""
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract medical entities from text (simplified version)"""
        # In production, this would use a proper NER model
        entities = []
        # Extract capitalized multi-word terms as potential entities
        matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        entities.extend(matches)
        
        # Extract known cardiology terms
        for term in self.cardiology_terms:
            if term.lower() in text.lower():
                entities.append(term)
        
        return list(set(entities))
    
    def _extract_medical_claims(self, text: str) -> List[str]:
        """Extract medical claims from text"""
        # Simple approach: sentences containing medical keywords and strong verbs
        medical_claims = []
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        claim_indicators = [
            r"(?i)studies show", r"(?i)research indicates", 
            r"(?i)proven to", r"(?i)evidence suggests",
            r"(?i)is effective", r"(?i)recommended for",
            r"(?i)should be used", r"(?i)must not",
            r"(?i)always causes", r"(?i)never causes"
        ]
        
        for sentence in sentences:
            if any(re.search(pattern, sentence) for pattern in claim_indicators):
                if any(term.lower() in sentence.lower() for term in self.cardiology_terms):
                    medical_claims.append(sentence)
        
        return medical_claims
    
    def _has_supporting_evidence(self, claim: str, context: str) -> bool:
        """Check if a claim has supporting evidence in the context"""
        # Extract key terms from the claim
        claim_lower = claim.lower()
        key_terms = [term for term in self.cardiology_terms if term.lower() in claim_lower]
        
        # Check if key terms and surrounding context appear in source
        if not key_terms:
            return True  # No specific medical terms to verify
            
        for term in key_terms:
            term_lower = term.lower()
            # Find nearby words in the claim
            term_idx = claim_lower.find(term_lower)
            if term_idx >= 0:
                # Get words around the term
                start = max(0, term_idx - 20)
                end = min(len(claim), term_idx + len(term) + 20)
                surrounding = claim[start:end].lower()
                
                # Check if this context appears in source
                if surrounding not in context.lower():
                    # Look for term itself at minimum
                    if term_lower not in context.lower():
                        return False
        
        return True
    
    def _compute_sentence_similarity(self, sentences: List[str]) -> float:
        """Compute average similarity between sentences (simplified)"""
        # In production, this would use embeddings and proper similarity metrics
        # Simplified version: count word overlap
        if len(sentences) <= 1:
            return 0
            
        word_sets = [set(s.lower().split()) for s in sentences]
        similarities = []
        
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if not word_sets[i] or not word_sets[j]:
                    continue
                    
                overlap = len(word_sets[i].intersection(word_sets[j]))
                similarity = overlap / min(len(word_sets[i]), len(word_sets[j]))
                similarities.append(similarity)
        
        return max(similarities) if similarities else 0
    
    def verify_medical_fact(self, statement: str) -> bool:
        """Verify a medical fact using knowledge-intensive approach"""
        response = self.medical_fact_checker(
            f"Verify this medical statement: {statement}",
            max_length=100
        )
        verification = response[0]['generated_text'].lower()
        return "true" in verification or "correct" in verification or "accurate" in verification
    
    def validate_answer(self, context: str, answer: str) -> Tuple[bool, str]:
        """
        Comprehensive validation of clinical answer for factuality,
        coherence and hallucination detection.
        
        Returns:
            tuple: (is_valid, message)
                - is_valid: Boolean indicating if the answer passed validation
                - message: Explanation of validation result or error
        """
        # Check for contradictions with context
        has_contradiction, contradiction_msg = self.check_contradiction(context, answer)
        if has_contradiction:
            return False, f"Contradiction detected: {contradiction_msg}"
        
        # Check for hallucinations
        has_hallucination, hallucination_msg = self.detect_hallucination(context, answer)
        if has_hallucination:
            return False, f"Hallucination detected: {hallucination_msg}"
        
        # Check for incoherence
        is_incoherent, incoherence_msg = self.check_incoherence(answer)
        if is_incoherent:
            return False, f"Incoherent answer: {incoherence_msg}"
        
        # Verify key medical facts (limit to first 2 to avoid performance issues)
        statements = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        for statement in statements[:2]:
            if not self.verify_medical_fact(statement):
                return False, f"Unverified medical claim: {statement}"
        
        return True, "Answer validated"

    def get_validation_flags(self, context: str, answer: str) -> Dict[str, Union[bool, str]]:
        """
        Get detailed validation flags for the answer.
        
        Returns:
            dict: Dictionary with validation flags and messages
                - is_valid: Overall validity
                - has_contradiction: Whether answer contradicts context
                - has_hallucination: Whether answer contains hallucinations
                - is_incoherent: Whether answer is incoherent
                - message: Overall validation message
        """
        has_contradiction, contradiction_msg = self.check_contradiction(context, answer)
        has_hallucination, hallucination_msg = self.detect_hallucination(context, answer)
        is_incoherent, incoherence_msg = self.check_incoherence(answer)
        
        is_valid = not (has_contradiction or has_hallucination or is_incoherent)
        message = "Answer validated"
        
        if not is_valid:
            if has_contradiction:
                message = f"Contradiction: {contradiction_msg}"
            elif has_hallucination:
                message = f"Hallucination: {hallucination_msg}"
            elif is_incoherent:
                message = f"Incoherence: {incoherence_msg}"
        
        return {
            "is_valid": is_valid,
            "has_contradiction": has_contradiction,
            "has_hallucination": has_hallucination,
            "is_incoherent": is_incoherent,
            "message": message
        }

    def format_validation_report(self, context: str, answer: str) -> str:
        """
        Generate a user-friendly validation report for the answer.
        
        Returns:
            str: Formatted validation report
        """
        flags = self.get_validation_flags(context, answer)
        
        report = []
        report.append("## Answer Validation Report\n")
        
        # Overall result
        if flags["is_valid"]:
            report.append("✅ **VALIDATED**: Answer appears factual and coherent\n")
        else:
            report.append("⚠️ **VALIDATION FAILED**: Issues detected in the answer\n")
        
        # Detailed checks
        report.append("### Validation Checks:\n")
        report.append(f"- Contradiction Check: {'❌ Failed' if flags['has_contradiction'] else '✅ Passed'}")
        report.append(f"- Hallucination Check: {'❌ Failed' if flags['has_hallucination'] else '✅ Passed'}")
        report.append(f"- Coherence Check: {'❌ Failed' if flags['is_incoherent'] else '✅ Passed'}")
        
        # Detailed message
        if not flags["is_valid"]:
            report.append(f"\n### Issue Details:\n{flags['message']}")
        
        return "\n".join(report)