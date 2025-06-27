import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATA_PROCESSED_KG_DIR = os.path.join(DATA_PROCESSED_DIR, 'kg')
DATA_PROCESSED_EMBEDDINGS_DIR = os.path.join(DATA_PROCESSED_DIR, 'embeddings')
DATA_PROCESSED_MODELS_DIR = os.path.join(DATA_PROCESSED_DIR, 'models')

# UMLS paths
UMLS_DIR = os.path.join(DATA_RAW_DIR, 'umls')
MRCONSO_PATH = os.path.join(UMLS_DIR, 'MRCONSO.RRF')
MRREL_PATH = os.path.join(UMLS_DIR, 'MRREL.RRF')
MRSTY_PATH = os.path.join(UMLS_DIR, 'MRSTY.RRF')

# SNOMED CT path
SNOMED_DIR = os.path.join(DATA_RAW_DIR, 'snomed_ct')

# DrugBank paths
DRUGBANK_PATH = os.path.join(DATA_RAW_DIR, 'drugbank.csv')
DRUGBANK_VOCAB_PATH = os.path.join(DATA_RAW_DIR, 'drugbank_vocabulary.csv')

# BioASQ path
BIOASQ_PATH = os.path.join(DATA_RAW_DIR, 'BioASQ', 'training13b.json')

# MedQuAD path
MEDQUAD_PATH = os.path.join(DATA_RAW_DIR, 'medquad', 'MedQuAD.csv')

# Model names
BIOBERT_MODEL = "dmis-lab/biobert-v1.1"
BIOGPT_MODEL = "microsoft/biogpt"

# Cardiology semantic types (TUI codes)
CARDIOLOGY_SEMANTIC_TYPES = {
    "T001", "T019", "T020", "T033", "T046", 
    "T047", "T048", "T121", "T184", "T200", "T201"
}