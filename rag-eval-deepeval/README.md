# rag-eval-deepeval

A RAG retrieval pipeline over YouTube course transcripts, evaluated with DeepEval, using Claude as the judge model instead of OpenAI.

## What this covers

### Retrieval pipeline (`src/retriever.py`)
- Loads `.vtt` transcripts from `data/`, strips timestamps, and chunks them with `RecursiveCharacterTextSplitter`.
- Embeds chunks locally with `BAAI/bge-base-en-v1.5` (via `sentence-transformers`) and stores them in a persistent Chroma vector store. No OpenAI embeddings are used.
- `answer(question)` retrieves the top-k chunks and generates a grounded answer with Claude Haiku, returning both the answer and the retrieval context for evaluation.

### Reranking (`src/reranker.py`)
- `RerankingRetriever` over-retrieves a larger candidate pool with the bi-encoder, then reorders it with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keeps the top-k.

### DeepEval + Claude integration
DeepEval's metrics and synthesizer default to OpenAI as the judge model. Since this project runs on Anthropic credits only, each script that needs a judge LLM defines a small `ClaudeJudge` class (`DeepEvalBaseLLM` subclass) backed by the `instructor` library, which forces schema-validated structured output out of Claude's plain messages API. This wrapper is used in:
- `evals/eval_retriever.py`
- `evals/eval_retriever_with_reranker.py`
- `goldens/generate_goldens.py`
- `resources/deepeval_intro.py`

### Evaluation (`evals/`)
- `eval_retriever.py` evaluates the base retriever with `ContextualRecallMetric` and `ContextualPrecisionMetric` against a golden set.
- `eval_retriever_with_reranker.py` runs the same evaluation against the reranked retriever, so the two can be compared directly.

### Goldens (`goldens/`)
- `retriever_goldens.json` — hand-authored question / ideal-answer pairs used as ground truth.
- `generate_goldens.py` — uses DeepEval's `Synthesizer` (via `ClaudeJudge`) to draft additional goldens from transcript chunks; output goes to `retriever_deepeval_goldens.json` and should be reviewed before use.

### Introductory example (`resources/deepeval_intro.py`)
- A minimal `AnswerRelevancyMetric` example showing a passing and a failing test case, judged by Claude.

## Results so far
Tuning the base retriever (larger `fetch_k`, switching to the `bge-base-en-v1.5` embedding model, and reducing chunk size from 1000 to 500) raised the reranked retriever's scores on the golden set:

| Metric | Before | After |
|---|---|---|
| Contextual Recall | 0.82 avg, 80% pass | 0.91 avg, 93% pass |
| Contextual Precision | 0.86 avg, 80% pass | 0.90 avg, 87% pass |

## Setup
```
uv sync
```
Create a `.env` file with:
```
ANTHROPIC_API_KEY=your-key-here
```

## Running it
```
python src/retriever.py
python -m evals.eval_retriever
python -m evals.eval_retriever_with_reranker
python goldens/generate_goldens.py
python resources/deepeval_intro.py
```
