# RAGCraft

**A comprehensive, hands-on implementation of Retrieval-Augmented Generation (RAG) — from foundational pipelines to advanced Corrective RAG, multimodal systems, and production database integrations.**

---

## Overview

RAGCraft is a structured learning repository that walks through the full spectrum of RAG techniques. It is organized into three main modules:

- **`corrective-rag/`** — A progressive, 6-notebook series implementing the [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884) paper step by step using LangChain + LangGraph.
- **`self-rag/`** — A progressive, 8-notebook series implementing the [Self-RAG](https://arxiv.org/abs/2310.11511) paper, teaching the model to reflect on its own retrieval and generation decisions using four critique signals: `Retrieve`, `IsREL`, `IsSUP`, and `IsUSE`.
- **`RAGify/`** — A broader collection covering RAG from scratch, database-backed retrieval (MongoDB, AstraDB, LanceDB), and multimodal RAG with Gemini, LlamaIndex, and more.

---

## Repository Structure

```
RAGCraft/
│
├── README.md
├── requirements.txt
│
├── corrective-rag/                         # Corrective RAG (CRAG) — step-by-step series
│   ├── 1_basic_rag.ipynb                   # Foundational RAG pipeline
│   ├── 2_retrieval_refinement.ipynb        # Sentence-level retrieval refinement
│   ├── 3_retrieval_evaluator.ipynb         # Document-level relevance scoring
│   ├── 4_web_search_refinement.ipynb       # Web search augmentation via Tavily
│   ├── 5_query_rewrite.ipynb               # Query rewriting for better retrieval
│   ├── 6_ambiguous.ipynb                   # Handling ambiguous / multi-intent queries
│   ├── Corrective-RAG paper.pdf            # Source research paper
│   ├── crag.png                            # CRAG architecture diagram
│   └── documents/                          # Sample PDFs used across notebooks
│       ├── book1.pdf
│       ├── book2.pdf
│       └── book3.pdf
│
├── self-rag/                               # Self-RAG — step-by-step series
│   ├── self_rag_step1.ipynb                # Adaptive retrieval decision (Retrieve token)
│   ├── self_rag_step2.ipynb                # Document relevance filtering (IsREL)
│   ├── self_rag_step3.ipynb                # Context-grounded generation
│   ├── self_rag_step4.ipynb                # Answer support verification (IsSUP)
│   ├── self_rag_step5.ipynb                # Retry limit & loop control
│   ├── self_rag_step6.ipynb                # Usefulness evaluation (IsUSE)
│   ├── self_rag_step7.ipynb                # Query rewriting — complete Self-RAG
│   ├── self_rag_web.ipynb                  # Bonus: web search variant
│   ├── self-rag paper.pdf                  # Source research paper
│   ├── self-rag.png                        # Self-RAG architecture diagram
│   └── documents/                          # NexaAI sample PDFs
│       ├── Company_Policies.pdf
│       ├── Company_Profile.pdf
│       └── Product_and_Pricing.pdf
│
└── RAGify/                                 # Diverse RAG implementations
    ├── requirements.txt
    └── RAG/
        ├── notebooks/
        │   └── RAG_Pipeline_from_Scratch.ipynb           # End-to-end RAG from first principles
        │
        ├── MongoDB and RAG/
        │   └── rag_with_huggingface_and_mongodb.ipynb    # HuggingFace embeddings + MongoDB
        │
        └── Multimodal RAG/
            ├── Chat_With_Multiple_Docs_AstraDB.ipynb         # Chat over PDFs/docs/pptx via AstraDB
            ├── Extract_Image_Table_Text_Summarizer_RAG.ipynb # Multimodal doc summarization
            ├── Extract_Image_Table_Text_document_parsing.ipynb  # Deep document parsing
            ├── Multimodal_RAG_Gemini.ipynb                   # RAG with Google Gemini
            └── multimodal_rag_llamaIndex_LanceDB.ipynb       # LlamaIndex + LanceDB
```

---

## Modules

### 1. Corrective RAG (CRAG) Series

Implements the CRAG paper as a 6-step progressive series. Each notebook builds on the previous, adding a new capability to the pipeline:

| # | Notebook | Concept Added |
|---|----------|---------------|
| 1 | `1_basic_rag.ipynb` | Basic retrieval + generation with FAISS |
| 2 | `2_retrieval_refinement.ipynb` | Sentence-level relevance filtering |
| 3 | `3_retrieval_evaluator.ipynb` | Document scoring with Pydantic + LangGraph |
| 4 | `4_web_search_refinement.ipynb` | Tavily web search fallback |
| 5 | `5_query_rewrite.ipynb` | Query rewriting for semantic alignment |
| 6 | `6_ambiguous.ipynb` | Ambiguous query detection and clarification |

**Stack:** `LangChain` · `LangGraph` · `FAISS` · `OpenAI` · `Tavily Search`

---

### 2. Self-RAG Series

Implements the Self-RAG paper ([arXiv 2310.11511](https://arxiv.org/abs/2310.11511)) as an 8-step progressive series. Each notebook builds on the previous, adding one new self-reflection capability to the LangGraph pipeline:

| # | Notebook | Concept Added | Self-RAG Token |
|---|----------|---------------|---------------|
| 1 | `self_rag_step1.ipynb` | Adaptive retrieval decision | `Retrieve` |
| 2 | `self_rag_step2.ipynb` | Document relevance filtering | `IsREL` |
| 3 | `self_rag_step3.ipynb` | Context-grounded generation | — |
| 4 | `self_rag_step4.ipynb` | Answer support verification + revise loop | `IsSUP` |
| 5 | `self_rag_step5.ipynb` | Retry limit & loop control | — |
| 6 | `self_rag_step6.ipynb` | Usefulness evaluation | `IsUSE` |
| 7 | `self_rag_step7.ipynb` | Query rewriting — complete Self-RAG | — |
| Bonus | `self_rag_web.ipynb` | Live web search variant | all |

**Stack:** `LangChain` · `LangGraph` · `FAISS` · `OpenAI` · `Pydantic`

---

### 3. RAG from Scratch

`RAGify/RAG/notebooks/RAG_Pipeline_from_Scratch.ipynb`

A 146-cell comprehensive walkthrough of every component in a RAG system — chunking, embedding, indexing, retrieval, and generation — built without abstractions to expose the underlying mechanics.

**Stack:** `sentence-transformers` · `FAISS` · `PyMuPDF` · `OpenAI`

---

### 4. Database-Backed RAG

`RAGify/RAG/MongoDB and RAG/rag_with_huggingface_and_mongodb.ipynb`

Production-style RAG using HuggingFace embeddings and MongoDB as a persistent vector store. Demonstrates how to move beyond in-memory FAISS to a scalable, server-backed retrieval system.

**Stack:** `transformers` · `sentence-transformers` · `pymongo` · `LangChain`

---

### 5. Multimodal RAG

Five notebooks covering different flavors of multimodal RAG:

| Notebook | Description |
|----------|-------------|
| `Chat_With_Multiple_Docs_AstraDB.ipynb` | Stateful chat over PDFs, docs, txt, pptx via AstraDB |
| `Extract_Image_Table_Text_Summarizer_RAG.ipynb` | Extract and summarize images, tables, and text from documents |
| `Extract_Image_Table_Text_document_parsing.ipynb` | Advanced parsing with the `unstructured` library |
| `Multimodal_RAG_Gemini.ipynb` | Native multimodal RAG using Google Gemini + LangChain |
| `multimodal_rag_llamaIndex_LanceDB.ipynb` | LlamaIndex orchestration with LanceDB vector store |

**Stack:** `unstructured` · `LangChain` · `Google Gemini` · `LlamaIndex` · `LanceDB` · `AstraDB`

---

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/HarshTomar1234/RAGCraft.git
cd RAGCraft
pip install -r requirements.txt
```

### API Keys Required

Create a `.env` file at the project root and populate the relevant keys for the notebooks you intend to run:

```env
OPENAI_API_KEY=your_openai_key

# For web search augmentation (corrective-rag notebooks 4–6)
TAVILY_API_KEY=your_tavily_key

# For Gemini multimodal RAG
GOOGLE_API_KEY=your_google_key

# For AstraDB (multimodal RAG)
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_API_ENDPOINT=your_astra_endpoint

# For MongoDB RAG
MONGODB_URI=your_mongodb_connection_string
```

> Not all keys are needed — each notebook only requires the services it uses.

---

## Learning Path

If you're new to RAG, follow this recommended progression:

```
1. corrective-rag/1_basic_rag.ipynb              ← Start here
2. RAGify/RAG/notebooks/RAG_Pipeline_from_Scratch.ipynb
3. corrective-rag/2_retrieval_refinement.ipynb
4. corrective-rag/3_retrieval_evaluator.ipynb
5. corrective-rag/4_web_search_refinement.ipynb
6. corrective-rag/5_query_rewrite.ipynb
7. corrective-rag/6_ambiguous.ipynb
8. RAGify/RAG/MongoDB and RAG/rag_with_huggingface_and_mongodb.ipynb
9. RAGify/RAG/Multimodal RAG/ (any order)        ← Advanced
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| LLM Orchestration | LangChain, LangGraph, LlamaIndex |
| LLMs | OpenAI GPT-4, Google Gemini |
| Embeddings | OpenAI, HuggingFace sentence-transformers |
| Vector Stores | FAISS, AstraDB, LanceDB, MongoDB |
| Document Parsing | PyMuPDF, pdfplumber, unstructured |
| Web Search | Tavily Search API |
| Validation | Pydantic |

---

## Reference

- [Corrective Retrieval Augmented Generation (CRAG) — arXiv 2401.15884](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
