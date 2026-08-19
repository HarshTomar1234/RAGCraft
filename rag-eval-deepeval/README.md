# rag-eval-proj

A RAG pipeline built over YouTube course transcripts (LLM evaluation lectures), evaluated end-to-end with DeepEval. Generation and judging both run on Claude — this project has no active OpenAI usage.

## What's implemented

### Retrieval (`src/retriever.py`)
- Loads `.vtt` transcripts from `data/`, strips timestamps, chunks them with `RecursiveCharacterTextSplitter` (500 chars, 100 overlap).
- Embeds chunks locally with `BAAI/bge-base-en-v1.5` via `sentence-transformers` and stores them in a persistent Chroma vector store (`chroma_store/`). No OpenAI embeddings.
- `answer(question)` retrieves the top-k chunks and generates a grounded answer with Claude Haiku directly via the `anthropic` SDK, returning both the answer and retrieval context.

### Generation (`src/generator.py`)
- The generator component in isolation: `generate(query, context) -> answer`.
- Faithfulness-first prompt — answers only from the given context, explicitly abstains when the context doesn't cover the question.
- Uses `ChatAnthropic` (`claude-haiku-4-5`) via LangChain's `prompt | llm | StrOutputParser` chain.

### Reranking (`src/reranker.py`)
- `RerankingRetriever`: over-retrieves a candidate pool with the bi-encoder (`fetch_k=20`), reorders with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), keeps the top-k.

### Full pipeline (`src/rag_pipeline.py`)
- `RagPipeline`: wires `RerankingRetriever` + `generate` into one call — retrieve → rerank → generate — returning `{query, context, answer}` for evaluation.

### DeepEval + Claude integration
DeepEval's metrics and `Synthesizer` default to OpenAI as the judge/generator model. Since this project runs on Anthropic credits only, every script that needs a judge LLM defines a `ClaudeJudge(DeepEvalBaseLLM)` wrapper backed by the `instructor` library, which forces schema-validated structured output out of Claude's plain messages API. Used in:
- `evals/eval_retriever.py`
- `evals/eval_retriever_with_reranker.py`
- `evals/eval_generator.py`
- `evals/eval_rag_pipeline.py`
- `goldens/generate_goldens.py`
- `resources/deepeval_intro.py`

**Model choice per eval:** `claude-haiku-4-5` is the default judge everywhere. `claude-sonnet-5` was tried as the judge for `eval_rag_pipeline.py`'s `ContextualRelevancyMetric`, but it hit a reproducible tool-call formatting bug on that metric's schema — it wraps output in a bogus `{'$PARAMETER_NAME': ..., '$PARAMETER_VALUE': ...}` shape instead of the expected schema, and once it locks onto that shape, retries don't recover it (confirmed after 9 retries with growing context). Haiku doesn't hit this. `eval_rag_pipeline.py` stays pinned to Haiku for that reason.

### Evaluation (`evals/`)
- `eval_retriever.py` — base retriever, `ContextualRecallMetric` + `ContextualPrecisionMetric`.
- `eval_retriever_with_reranker.py` — same metrics against the reranked retriever, for direct comparison.
- `eval_generator.py` — generator in isolation (fixed golden context, not retriever output), `FaithfulnessMetric` + `AnswerRelevancyMetric`.
- `eval_rag_pipeline.py` — full pipeline on live output, `ContextualRelevancyMetric` + `FaithfulnessMetric` + `AnswerRelevancyMetric`.

### Goldens (`goldens/`)
- `retriever_goldens.json` — hand-authored question / ideal-answer pairs for retriever eval.
- `faithfulness_dataset.json` — query + ideal-context pairs for isolated generator eval.
- `generate_goldens.py` — uses DeepEval's `Synthesizer` (via `ClaudeJudge`) to draft additional goldens from transcript chunks; output goes to `retriever_deepeval_goldens.json` and should be reviewed before use.

### Other
- `resources/deepeval_intro.py` — minimal `AnswerRelevancyMetric` example (pass/fail case), judged by Claude.
- `export_chroma_chunks.py` — dumps every stored chunk (id, text, metadata) from Chroma to JSON, for hand-picking golden context.

## Results so far

Tuning the base retriever — larger `fetch_k` (10→20), switching to the `bge-base-en-v1.5` embedding model, reducing chunk size (1000→500) — raised the reranked retriever's scores:

| Metric | Before | After |
|---|---|---|
| Contextual Recall | 0.82 avg, 80% pass | 0.91 avg, 93% pass |
| Contextual Precision | 0.86 avg, 80% pass | 0.90 avg, 87% pass |

Full pipeline eval (`eval_rag_pipeline.py`, Haiku judge):

| Metric | Avg Score | Pass Rate |
|---|---|---|
| Contextual Relevancy | 0.76 | 80% (12/15) |
| Faithfulness | 0.98–0.99 | 100% (15/15) |
| Answer Relevancy | 0.92–0.95 | 87% (13/15) |

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
python src/generator.py
python -m src.rag_pipeline
python -m evals.eval_retriever
python -m evals.eval_retriever_with_reranker
python -m evals.eval_generator
python -m evals.eval_rag_pipeline
python goldens/generate_goldens.py
python resources/deepeval_intro.py
```
