# CARDIO-LR System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          User Interface                            │
│  ┌─────────────┐                                   ┌────────────┐  │
│  │ Query Input │                                   │ Answer UI  │  │
│  └─────────────┘                                   └────────────┘  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       CardiologyLightRAG Pipeline                  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Patient Context Processor                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Entity Extraction → Medical Entity Linking → Context Model  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Hybrid Retriever                             │
│  ┌───────────────────┐                     ┌────────────────────────┐ │
│  │ Vector Retrieval  │◄───┐           ┌───►│ Knowledge Graph Query  │ │
│  │ (FAISS + BioBERT) │    │           │    │ (Symbolic Retrieval)   │ │
│  └───────────────────┘    │           │    └────────────────────────┘ │
│                           │           │                               │
│                      ┌────┴───────────┴────┐                          │
│                      │   Fusion Component  │                          │
│                      └────────────────────┬┘                          │
└───────────────────────────────────────────┼──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Subgraph Extractor                             │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ Graph Neural Network → Relevant Subgraph → Text Conversion    │   │
│  └───────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Context Integration                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Patient Context + Retrieved Knowledge + Subgraph Knowledge      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Biomedical Generator                            │
│  ┌───────────────────┐                     ┌────────────────────────┐ │
│  │ BioGPT Model      │                     │ Answer Formatting      │ │
│  └───────────────────┘                     └────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Answer Validator                              │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ Clinical Guideline Check → Claim Verification → Source Check  │   │
│  └───────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Explainability Module                            │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ Reasoning Trace → Evidence Linking → Clinical Report          │   │
│  └───────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
                      ┌─────────────────────┐
                      │  Clinical Answer    │
                      │  with Explanation   │
                      └─────────────────────┘
```

## Data Flow

1. User submits a cardiology question with optional patient context
2. Patient context is processed to extract medical entities and conditions
3. Hybrid retrieval combines vector search and knowledge graph querying
4. Subgraph extractor identifies relevant portions of the medical knowledge graph
5. Context integration combines patient information with retrieved knowledge
6. Biomedical generator produces a draft clinical answer
7. Answer validator ensures clinical accuracy and safety
8. Explainability module generates reasoning report
9. Final answer with explanation is presented to the user

## Key Components

### Patient Context Processor
- Uses ScispaCy for biomedical NER
- Links extracted entities to UMLS concepts
- Builds a patient-specific context model

### Hybrid Retriever
- Vector component uses BioGPT embeddings with FAISS indexing
- Symbolic component performs graph traversal queries on the knowledge graph
- Fusion combines results based on clinical relevance scoring

### Subgraph Extractor
- Uses R-GCN (Relational Graph Convolutional Network) to identify relevant subgraphs
- Performs multi-hop reasoning over medical knowledge
- Converts subgraphs to natural language for the generator

### Biomedical Generator
- Fine-tuned BioGPT model for clinical question answering
- Prompt engineering includes retrieved evidence and patient context
- Structured output format for clinical answers

### Answer Validator
- Verifies generated content against medical guidelines
- Checks factual consistency with source knowledge
- Ensures answers address the original clinical query

### Explainability Module
- Traces the reasoning path from knowledge to answer
- Links evidence to specific claims in the generated answer
- Provides a clinical reasoning report for transparency