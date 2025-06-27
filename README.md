# CARDIO-LR: Cardiology Knowledge-Enhanced LightRAG System
# Setup and Usage Guide

## Project Overview

CARDIO-LR is a specialized question answering system for the cardiology domain that enhances the LightRAG (Retrieval-Augmented Generation) approach with medical knowledge graphs. The system integrates information from UMLS, SNOMED CT, and DrugBank with vector-based retrieval and uses Bio_ClinicalBERT for improved medical language understanding.

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- 16GB+ RAM
- 50GB+ disk space (for medical knowledge bases)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CARDIO-LR.git
cd CARDIO-LR
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Library Installation

```bash
python verify_libraries.py
```

### 5. Download and Prepare Datasets

#### 5.1 Download Required Datasets

- **BioASQ**: Download from the official BioASQ website (https://bioasq.org/) and place in `data/raw/BioASQ/`
- **MedQuAD**: Download from https://github.com/abachaa/MedQuAD and place in `data/raw/medquad/`
- **UMLS**: Access requires UMLS license from https://www.nlm.nih.gov/research/umls/ - place RRF files in `data/raw/umls/`
- **SNOMED CT**: Access requires license from https://www.snomed.org/ - place snapshot files in `data/raw/snomed_ct/`
- **DrugBank**: Download from https://go.drugbank.com/ (requires registration) - place CSV files in `data/raw/drugbank/`

#### 5.2 Filter Datasets for Cardiology Content

```bash
# Filter BioASQ and MedQuAD datasets for cardiology content
python kg_construction/data_filter.py
```

### 6. Build Knowledge Graphs

```bash
# Process UMLS data and extract cardiology-specific knowledge
python kg_construction/umls_processor.py

# Process SNOMED CT data
python kg_construction/snomed_processor.py

# Process DrugBank data
python kg_construction/drugbank_processor.py

# Integrate all knowledge graphs
python kg_construction/knowledge_integrator.py
```

### 7. Generate Embeddings and FAISS Index

```bash
# Generate embeddings and build retrieval index
python retrieval/embedding_generator.py
python retrieval/faiss_indexer.py
```

## Running the System

### Simple Command Line Demo

```bash
python pipeline.py "What are the first-line treatments for atrial fibrillation in elderly patients with renal impairment?"
```

### Interactive Demo

```bash
python run_demo.py
```

This will start an interactive demo where you can:
- Enter cardiology questions
- Provide patient context information
- View answers with explanations
- See the system's retrieval and reasoning process

### Jupyter Notebook

For more detailed exploration and visualization:

```bash
jupyter notebook notebooks/system_demonstration.ipynb
```

## Running Evaluation

To evaluate the system and compare it with the baseline LightRAG:

```bash
python evaluation/evaluate.py
```

This will:
1. Run a comprehensive evaluation on test questions
2. Generate performance metrics (EM, F1, ROUGE-L, medical accuracy)
3. Create comparative visualizations
4. Produce an evaluation report

Results will be saved in the `benchmark_results/` directory.

## System Architecture

CARDIO-LR follows a multi-stage pipeline:

1. **Query Processing**: Parse the user's cardiology question and any patient context
2. **Retrieval**: Hybrid retrieval combining dense vector search with symbolic knowledge graph retrieval
3. **Knowledge Graph Integration**: Extract a relevant subgraph from medical knowledge bases
4. **Context Enhancement**: Use Bio_ClinicalBERT to enhance understanding and link text to knowledge graph
5. **Personalization**: Integrate patient-specific context with medical knowledge
6. **Answer Generation**: Generate a comprehensive answer using all available information
7. **Validation**: Validate the medical accuracy of the generated answer
8. **Explanation**: Provide an explanation of the reasoning process

## Key Components

### 1. Knowledge Graph Construction

Built from three main sources:
- **UMLS**: Unified Medical Language System - provides comprehensive medical terminology
- **SNOMED CT**: Clinical terminology with detailed cardiology concepts
- **DrugBank**: Medication information including interactions and contraindications

### 2. Retrieval System

- **Vector-based Retrieval**: Uses FAISS (Facebook AI Similarity Search) with sentence transformer embeddings
- **Knowledge Graph Retrieval**: Symbolic retrieval using medical concept mapping
- **Hybrid Approach**: Combines both methods for comprehensive information retrieval

### 3. Medical Language Understanding

- **Bio_ClinicalBERT**: Specialized BERT model fine-tuned on clinical text
- **Medical Term Recognition**: Identifies cardiology-specific terms in queries
- **Knowledge Graph Linking**: Maps text mentions to knowledge graph concepts

### 4. Personalization

- **Patient Context Processing**: Extracts relevant medical information from patient descriptions
- **Context Integration**: Relates patient-specific factors to general medical knowledge
- **Personalized Answers**: Tailors responses based on patient characteristics

## Example Questions

The system is optimized for cardiology questions such as:

1. "What are the first-line treatments for stable angina in diabetic patients?"
2. "How should beta-blockers be used in patients with heart failure?"
3. "What are the complications of atrial fibrillation?"
4. "Is aspirin safe for a patient with a history of GI bleeding who has coronary artery disease?"
5. "What antihypertensive medications are appropriate for a diabetic patient with heart failure?"

## Troubleshooting

### Common Issues

1. **Missing Knowledge Graph Files**: Ensure all KG files are properly generated in `data/processed/kg/`
   ```bash
   ls -la data/processed/kg/
   ```

2. **CUDA Out of Memory**: Reduce batch sizes in `generation/clinical_bert_enhancer.py`

3. **Missing Dependencies**: If you encounter missing library errors:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

4. **FAISS Index Load Error**: Rebuild the FAISS index:
   ```bash
   python retrieval/faiss_indexer.py --rebuild
   ```

### Support

For additional support, please report issues on the GitHub repository or contact the development team.

## Citation and References

If using this system for research, please cite:

```
@misc{CARDIO-LR2025,
  title={CARDIO-LR: A Cardiology-Specific LightRAG System with Knowledge Graph Integration},
  author={Team-3: Anuradha Chavan, Ankit Sharma, Khyati Maddali, Shweta Kirave},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/ASKAAI/CardioLightRAG}}
}
```

Key references:
- LightRAG: https://github.com/conceptofmind/LightRAG
- Bio_ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT 
- FAISS: https://github.com/facebookresearch/faiss
- UMLS: https://www.nlm.nih.gov/research/umls/
- SNOMED CT: https://www.snomed.org/

## License

This project is licensed under the terms of the MIT license.
