# RAGCraft

**A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations — from foundational pipelines to multimodal systems and production database integrations.**

---

## Overview

RAGCraft covers the full spectrum of modern RAG techniques through hands-on Jupyter notebooks. Each notebook is self-contained and progressively introduces new concepts, tools, and architectures.

**What's inside:**

- End-to-end RAG pipeline built from first principles
- Production-ready RAG with HuggingFace embeddings and MongoDB
- Multimodal RAG across PDFs, images, tables, and multiple document formats
- Integrations with AstraDB, LanceDB, Google Gemini, and LlamaIndex

---

## Repository Structure

```
RAGCraft/
├── README.md
├── requirements.txt
│
└── RAG/
    ├── notebooks/
    │   └── RAG_Pipeline_from_Scratch.ipynb
    │
    ├── MongoDB and RAG/
    │   └── rag_with_huggingface_and_mongodb.ipynb
    │
    └── Multimodal RAG/
        ├── Chat_With_Multiple_Docs(pdfs,_docs,_txt,_pptx)_using_AstraDB_and_Langchain.ipynb
        ├── Extract_Image,Table,Text_from_Document_MultiModal_Summarizer_RAG_App.ipynb
        ├── Extract_Image,Table,Text_from_document_parsing.ipynb
        ├── Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio.ipynb
        └── multimodal_rag_with_llamaIndex_&_LanceDB.ipynb
```

---

## Notebooks

### RAG from Scratch

**`RAG/notebooks/RAG_Pipeline_from_Scratch.ipynb`**

A deep-dive, 146-cell walkthrough of every component in a RAG system, built without high-level abstractions. Covers:

- Document loading and chunking strategies
- Text embedding with `sentence-transformers`
- FAISS-based vector indexing and similarity search
- Context injection and prompt construction
- Response generation and evaluation

**Stack:** `sentence-transformers` · `FAISS` · `PyMuPDF` · `transformers` · `OpenAI`

---

### RAG with HuggingFace and MongoDB

**`RAG/MongoDB and RAG/rag_with_huggingface_and_mongodb.ipynb`**

Production-style RAG replacing in-memory FAISS with a persistent MongoDB vector store and open-source HuggingFace embeddings instead of OpenAI. Demonstrates:

- HuggingFace embedding models as a drop-in OpenAI replacement
- MongoDB Atlas Vector Search for scalable retrieval
- End-to-end pipeline with persistent storage

**Stack:** `transformers` · `sentence-transformers` · `pymongo` · `LangChain`

---

### Multimodal RAG

Five notebooks covering different multimodal RAG patterns:

#### Chat with Multiple Document Types — AstraDB
**`RAG/Multimodal RAG/Chat_With_Multiple_Docs(pdfs,_docs,_txt,_pptx)_using_AstraDB_and_Langchain.ipynb`**

Stateful conversational RAG over heterogeneous document formats (PDF, DOCX, TXT, PPTX) using AstraDB as the vector backend. Supports multi-turn conversation with document context.

**Stack:** `LangChain` · `AstraDB` · `OpenAI` · `unstructured`

---

#### Multimodal Document Summarization
**`RAG/Multimodal RAG/Extract_Image,Table,Text_from_Document_MultiModal_Summarizer_RAG_App.ipynb`**

Extracts images, tables, and text from documents separately, generates summaries for each modality using a vision-capable LLM, then indexes all summaries for unified retrieval.

**Stack:** `unstructured` · `LangChain` · `OpenAI GPT-4V`

---

#### Document Parsing Deep Dive
**`RAG/Multimodal RAG/Extract_Image,Table,Text_from_document_parsing.ipynb`**

Focused exploration of the `unstructured` library for parsing complex document layouts — extracting structured tables, embedded images, and clean text blocks from real-world PDFs.

**Stack:** `unstructured` · `pdfplumber` · `PyMuPDF`

---

#### Multimodal RAG with Google Gemini
**`RAG/Multimodal RAG/Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio.ipynb`**

Leverages Google Gemini's native multimodal capabilities for RAG over documents containing text and images. Gemini handles both embedding and vision-based generation natively.

**Stack:** `LangChain` · `langchain-google-genai` · `Google AI Studio` · `Gemini 1.5`

---

#### Multimodal RAG with LlamaIndex and LanceDB
**`RAG/Multimodal RAG/multimodal_rag_with_llamaIndex_&_LanceDB.ipynb`**

Uses LlamaIndex for orchestration and LanceDB as the vector store — both optimized for multimodal data. Handles images, video frames, and text within a single retrieval pipeline.

**Stack:** `llama-index` · `LanceDB` · `OpenAI` · `sentence-transformers`

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

### API Keys

Create a `.env` file in the project root and populate the keys required by the notebooks you plan to run:

```env
OPENAI_API_KEY=your_openai_key

# MongoDB RAG
MONGODB_URI=your_mongodb_connection_string

# Multimodal RAG — Gemini
GOOGLE_API_KEY=your_google_api_key

# Multimodal RAG — AstraDB
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
```

> Each notebook only uses the keys relevant to its stack — you don't need all of them.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| LLM Orchestration | LangChain, LlamaIndex |
| LLMs | OpenAI GPT-4, Google Gemini 1.5 |
| Embeddings | OpenAI, HuggingFace sentence-transformers |
| Vector Stores | FAISS, MongoDB Atlas, AstraDB, LanceDB |
| Document Parsing | PyMuPDF, pdfplumber, unstructured |
| Data Validation | Pydantic |

---

## Related

- [Corrective RAG (CRAG) Series](https://github.com/campusx-official/corrective-rag) — A companion 6-notebook series implementing the CRAG paper with LangGraph.
