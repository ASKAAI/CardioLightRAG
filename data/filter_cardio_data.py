#!/usr/bin/env python3
"""
Dataset Filtering Script for Cardiology Questions

This script filters the BioASQ and MedQuAD datasets to extract cardiology-specific
questions based on a comprehensive list of cardiology-related keywords.

Usage:
    python filter_cardio_data.py

Output:
    - Filtered cardiology question-answer pairs from BioASQ
    - Filtered cardiology question-answer pairs from MedQuAD
    - Combined filtered dataset
    - Summary statistics
"""

import os
import pandas as pd
import json
import re
from tqdm import tqdm
import sys
import argparse

# Add project root to path for importing config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR

# Define paths (use default paths from config if available, otherwise use these)
BIOASQ_PATH = os.path.join(DATA_RAW_DIR, 'bioasq', 'training_data.json')
MEDQUAD_PATH = os.path.join(DATA_RAW_DIR, 'medquad', 'MedQuAD.csv')

# Define cardiology-related keywords for filtering
CARDIOLOGY_KEYWORDS = [
    # Cardiovascular conditions
    "heart", "cardiac", "cardiovascular", "cardio", "coronary",
    "angina", "myocardial infarction", "heart attack", "heart failure",
    "arrhythmia", "atrial fibrillation", "afib", "tachycardia",
    "bradycardia", "palpitation", "hypertension", "high blood pressure",
    "hypotension", "low blood pressure", "stroke", "tia", "thrombosis",
    "embolism", "aneurysm", "atherosclerosis", "arteriosclerosis",
    "valve", "mitral", "aortic", "tricuspid", "pulmonary hypertension",
    
    # Cardiovascular procedures
    "angiogram", "angioplasty", "stent", "cabg", "bypass", 
    "coronary bypass", "pacemaker", "defibrillator", "icd", 
    "ablation", "cardioversion", "ecg", "ekg", "echocardiogram",
    
    # Cardiovascular medications
    "anticoagulant", "antiplatelet", "aspirin", "warfarin", "heparin",
    "statin", "beta blocker", "ace inhibitor", "arb", "calcium channel blocker",
    "digoxin", "diuretic", "nitroglycerin", "nitrate",
    
    # Cardiology specialties & anatomy
    "cardiologist", "cardiology", "electrophysiology", "interventional",
    "aorta", "ventricle", "atrium", "myocardium", "endocardium", "pericardium",
    "septum", "conduction", "sinoatrial", "atrioventricular"
]

def contains_cardio_keyword(text):
    """Check if text contains any cardiology-related keywords"""
    if not isinstance(text, str):
        return False
        
    text_lower = text.lower()
    
    # Check for whole word matches to avoid partial matches (e.g., "art" in "article")
    for keyword in CARDIOLOGY_KEYWORDS:
        # For multi-word keywords
        if " " in keyword:
            if keyword in text_lower:
                return True
        # For single-word keywords, ensure whole word matching
        else:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return True
    
    return False

def filter_bioasq(bioasq_path):
    """Filter BioASQ dataset for cardiology questions"""
    print(f"Processing BioASQ dataset from {bioasq_path}")
    
    if not os.path.exists(bioasq_path):
        print(f"Warning: BioASQ file not found at {bioasq_path}")
        print("Creating placeholder data with example questions...")
        # Create some example data for demonstration
        bioasq_data = [
            {"body": "What are the mechanisms of action of beta-blockers in heart failure?", 
             "ideal_answer": "Beta-blockers work by blocking the effects of adrenaline..."},
            {"body": "How do statins reduce cardiovascular risk?",
             "ideal_answer": "Statins lower cholesterol production in the liver..."},
            # Add more examples as needed
        ]
    else:
        # Load BioASQ data
        try:
            with open(bioasq_path, 'r') as f:
                bioasq_data = json.load(f)
        except Exception as e:
            print(f"Error loading BioASQ data: {e}")
            bioasq_data = []
    
    # Filter for cardiology-related questions
    cardio_qa_pairs = []
    for qa_pair in tqdm(bioasq_data, desc="Filtering BioASQ"):
        question = qa_pair.get('body', '')
        if contains_cardio_keyword(question):
            cardio_qa_pairs.append({
                'question': question,
                'answer': qa_pair.get('ideal_answer', ''),
                'source': 'BioASQ'
            })
    
    print(f"Extracted {len(cardio_qa_pairs)} cardiology QA pairs from BioASQ")
    return cardio_qa_pairs

def filter_medquad(medquad_path):
    """Filter MedQuAD dataset for cardiology questions"""
    print(f"Processing MedQuAD dataset from {medquad_path}")
    
    if not os.path.exists(medquad_path):
        print(f"Warning: MedQuAD file not found at {medquad_path}")
        print("Creating placeholder data with example questions...")
        # Create some example data for demonstration
        medquad_data = pd.DataFrame({
            'Question': [
                "What are the first-line treatments for stable angina?",
                "What are the warning signs of a heart attack?",
                "Can hypertension cause kidney damage?",
                # Add more examples as needed
            ],
            'Answer': [
                "First-line treatments include beta-blockers, calcium channel blockers...",
                "Warning signs include chest pain, shortness of breath...",
                "Yes, hypertension is a leading cause of kidney damage...",
            ]
        })
    else:
        # Load MedQuAD data
        try:
            medquad_data = pd.read_csv(medquad_path)
        except Exception as e:
            print(f"Error loading MedQuAD data: {e}")
            # Create empty DataFrame with expected columns
            medquad_data = pd.DataFrame(columns=['Question', 'Answer'])
    
    # Filter for cardiology-related questions
    cardio_qa_pairs = []
    for _, row in tqdm(medquad_data.iterrows(), total=len(medquad_data), desc="Filtering MedQuAD"):
        try:
            question = row.get('Question', '')
            answer = row.get('Answer', '')
            
            # Check both question and answer for cardiology keywords
            if contains_cardio_keyword(question) or contains_cardio_keyword(answer):
                cardio_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': 'MedQuAD'
                })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Extracted {len(cardio_qa_pairs)} cardiology QA pairs from MedQuAD")
    return cardio_qa_pairs

def analyze_dataset(qa_pairs, dataset_name):
    """Analyze the filtered dataset and generate statistics"""
    print(f"\nAnalyzing {dataset_name} dataset...")
    
    # Count occurrences of each keyword
    keyword_counts = {keyword: 0 for keyword in CARDIOLOGY_KEYWORDS}
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '').lower()
        answer = qa_pair.get('answer', '').lower()
        
        # Count keyword occurrences
        for keyword in CARDIOLOGY_KEYWORDS:
            if " " in keyword:
                # Multi-word keywords
                if keyword in question or keyword in answer:
                    keyword_counts[keyword] += 1
            else:
                # Single-word keywords
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, question) or re.search(pattern, answer):
                    keyword_counts[keyword] += 1
    
    # Get top keywords
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 most frequent cardiology keywords in {dataset_name}:")
    for keyword, count in top_keywords:
        if count > 0:  # Only show keywords that appear at least once
            print(f"  - '{keyword}': {count} occurrences")
    
    return {
        'total_pairs': len(qa_pairs),
        'top_keywords': top_keywords
    }

def main(args):
    """Main function to filter and save cardiology QA pairs"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(DATA_PROCESSED_DIR, 'cardio_qa')
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter BioASQ dataset
    bioasq_cardio = filter_bioasq(args.bioasq or BIOASQ_PATH)
    
    # Filter MedQuAD dataset
    medquad_cardio = filter_medquad(args.medquad or MEDQUAD_PATH)
    
    # Combine the datasets
    all_cardio_qa = bioasq_cardio + medquad_cardio
    
    # Save the filtered datasets
    with open(os.path.join(output_dir, 'bioasq_cardio.json'), 'w') as f:
        json.dump(bioasq_cardio, f, indent=2)
    
    with open(os.path.join(output_dir, 'medquad_cardio.json'), 'w') as f:
        json.dump(medquad_cardio, f, indent=2)
        
    with open(os.path.join(output_dir, 'all_cardio_qa.json'), 'w') as f:
        json.dump(all_cardio_qa, f, indent=2)
    
    # Analyze the datasets
    bioasq_stats = analyze_dataset(bioasq_cardio, "BioASQ")
    medquad_stats = analyze_dataset(medquad_cardio, "MedQuAD")
    
    # Generate a summary file
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CARDIOLOGY DATASET FILTERING SUMMARY\n")
        f.write("===================================\n\n")
        f.write(f"BioASQ cardiology QA pairs: {len(bioasq_cardio)}\n")
        f.write(f"MedQuAD cardiology QA pairs: {len(medquad_cardio)}\n")
        f.write(f"Total cardiology QA pairs: {len(all_cardio_qa)}\n\n")
        
        f.write("FILTERING DETAILS\n")
        f.write("================\n\n")
        f.write(f"Cardiology keywords used for filtering: {len(CARDIOLOGY_KEYWORDS)}\n")
        f.write("Categories: conditions, procedures, medications, specialties/anatomy\n\n")
        f.write("TOP KEYWORDS IN BIOASQ\n")
        for keyword, count in bioasq_stats['top_keywords']:
            if count > 0:
                f.write(f"  - '{keyword}': {count} occurrences\n")
        
        f.write("\nTOP KEYWORDS IN MEDQUAD\n")
        for keyword, count in medquad_stats['top_keywords']:
            if count > 0:
                f.write(f"  - '{keyword}': {count} occurrences\n")
        
    print(f"\nTotal cardiology QA pairs: {len(all_cardio_qa)}")
    print(f"Data saved to {output_dir}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter medical datasets for cardiology questions')
    parser.add_argument('--bioasq', help='Path to BioASQ dataset file')
    parser.add_argument('--medquad', help='Path to MedQuAD dataset file')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during processing')
    
    args = parser.parse_args()
    main(args)