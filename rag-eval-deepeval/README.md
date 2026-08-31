# rag-eval-deepeval

**A production-oriented RAG evaluation system for measuring retrieval quality, generation quality, application behavior, and safety.**

This project builds and evaluates a complete Retrieval-Augmented Generation (RAG) pipeline over YouTube course transcripts covering LLM evaluation.

It goes beyond measuring whether a RAG system can retrieve relevant documents and generate faithful answers. The system evaluates the pipeline at multiple layers:

* **Retrieval quality**
* **Reranking effectiveness**
* **Generation quality**
* **End-to-end RAG quality**
* **Application correctness and completeness**
* **Response style**
* **Toxicity**
* **Prompt leakage**
* **Course-content leakage**
* **PII leakage**
* **Scope adherence**
* **Interactive application behavior**

Generation and evaluation are currently powered by **Claude**, with local BGE embeddings and a cross-encoder reranker.

---

## Overview

The project treats RAG evaluation as a layered engineering problem rather than a single score.

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   BGE Embeddings    │
                         │ bge-base-en-v1.5    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Chroma Retrieval  │
                         │     fetch_k = 20    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │     Reranking       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Claude Generator  │
                         │   Faithfulness-first│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Final Answer     │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
          ┌───────────────────┐             ┌──────────────────┐
          │ RAG Quality Evals │             │ Safety / Behavior│
          └─────────┬─────────┘             │      Evals       │
                    │                       └─────────┬────────┘
                    ▼                                 ▼
        ┌────────────────────────┐       ┌─────────────────────────┐
        │ Recall / Precision     │       │ Correctness             │
        │ Relevancy              │       │ Completeness            │
        │ Faithfulness           │       │ Style                   │
        │ Answer Relevancy       │       │ Toxicity                │
        └────────────────────────┘       │ Prompt Leakage          │
                                         │ Course Leakage          │
                                         │ PII Leakage             │
                                         │ Scope Adherence         │
                                         └─────────────────────────┘
```

The system therefore evaluates both:

> **"Did the RAG pipeline retrieve and answer correctly?"**

and:

> **"Does the resulting application behave safely and stay within its intended role?"**

---

# What is implemented

## 1. Retrieval

### `src/retriever.py`

The retrieval layer:

1. Loads `.vtt` YouTube transcripts.
2. Removes timestamp information.
3. Splits transcripts using `RecursiveCharacterTextSplitter`.
4. Uses:

   * chunk size: **500 characters**
   * overlap: **100 characters**
5. Generates local embeddings using:

```text
BAAI/bge-base-en-v1.5
```

6. Stores embeddings in a persistent **Chroma** vector store.
7. Retrieves the most relevant chunks for a query.
8. Passes the retrieved context to the Claude generator.

No OpenAI embeddings are required.

---

## 2. Reranking

### `src/reranker.py`

The system uses a two-stage retrieval strategy.

### Stage 1 — Bi-encoder retrieval

The BGE embedding model retrieves an expanded candidate pool:

```text
fetch_k = 20
```

### Stage 2 — Cross-encoder reranking

The candidate pool is reordered using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The final top-k documents are then passed to the generator.

```text
Query
  │
  ▼
BGE retrieval
  │
  │ 20 candidates
  ▼
Cross-encoder
  │
  │ reranked
  ▼
Top-k context
```

This makes it possible to evaluate the effect of reranking independently from the base retriever.

---

# 3. Generation

### `src/generator.py`

The generator can also be evaluated independently of retrieval.

```python
generate(query, context) -> answer
```

The generation prompt is designed around **grounded answering**.

The generator is instructed to:

* answer from the supplied context
* avoid unsupported claims
* handle partial context coverage explicitly
* answer the portion that can be supported
* acknowledge missing information when necessary
* avoid unnecessary padding
* avoid repetitive answers
* remain within the intended teaching-assistant scope
* avoid exposing protected instructions or sensitive information
* avoid inappropriate or toxic responses

Generation uses Claude through LangChain's `ChatAnthropic` integration.

---

# 4. Full RAG Pipeline

### `src/rag_pipeline.py`

`RagPipeline` combines retrieval, reranking, and generation:

```text
Query
  ↓
Retrieve
  ↓
Rerank
  ↓
Generate
  ↓
{
    query,
    context,
    answer
}
```

This component is used by the end-to-end evaluation suite.

---

# 5. Interactive Application

The project also includes a **Streamlit chat interface** for interacting with the RAG system.

This makes it possible to evaluate the system not only as isolated Python components, but as an actual user-facing application.

The application exercises the same live pipeline used by the end-to-end evaluations:

```text
User
 ↓
Streamlit UI
 ↓
RagPipeline
 ↓
Retrieval
 ↓
Reranking
 ↓
Claude
 ↓
Response
```

This provides a practical surface for manually testing:

* normal course questions
* partially supported questions
* unrelated requests
* adversarial prompts
* prompt extraction attempts
* content extraction attempts
* PII requests
* jailbreak / role-switch attempts

---

# Evaluation Architecture

The project deliberately evaluates different layers separately.

This makes it possible to identify **where** a RAG system fails rather than reducing the entire application to one aggregate score.

```text
                    RAG Evaluation
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Retriever       Generator       Application
          │               │                │
          ▼               ▼                ▼
    Recall/Precision  Faithfulness     Correctness
    Contextual       Answer            Completeness
    Relevancy        Relevancy          Style
                                            │
                                            ▼
                                      Safety / Security
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                          Toxicity      Leakage        Scope
```

DeepEval's current guidance similarly recommends evaluating RAG retrievers and generators separately as well as evaluating the complete system end-to-end.

---

# Evaluation Suite

## Retriever Evaluation

### `evals/eval_retriever.py`

Evaluates the base retriever using:

* `ContextualRecallMetric`
* `ContextualPrecisionMetric`

The purpose is to determine whether retrieval provides:

* sufficient relevant information
* focused and useful context

---

## Reranked Retriever Evaluation

### `evals/eval_retriever_with_reranker.py`

Runs the same retrieval metrics against the reranked pipeline.

This enables a direct comparison:

```text
Base Retriever
       vs
Retriever + Cross-Encoder
```

This was used during tuning of:

* embedding model
* chunk size
* retrieval candidate pool

---

# Generator Evaluation

### `evals/eval_generator.py`

The generator is evaluated independently using fixed golden contexts.

Metrics:

* `FaithfulnessMetric`
* `AnswerRelevancyMetric`

This isolates generation quality from retrieval quality.

That distinction is important because a poor final answer can originate from either:

```text
Bad Context
     OR
Bad Generation
```

Component-level evaluation makes those failure modes easier to distinguish.

---

# End-to-End RAG Evaluation

### `evals/eval_rag_pipeline.py`

Runs the live RAG pipeline and evaluates the generated result.

Metrics:

* `ContextualRelevancyMetric`
* `FaithfulnessMetric`
* `AnswerRelevancyMetric`

This provides the core RAG quality view:

| Dimension            | Question                                          |
| -------------------- | ------------------------------------------------- |
| Contextual Relevancy | Was the retrieved context useful?                 |
| Faithfulness         | Is the answer supported by the retrieved context? |
| Answer Relevancy     | Does the answer actually address the question?    |

These correspond closely to the standard RAG evaluation dimensions recommended by DeepEval.

---

# Application-Level Evaluation

### `evals/eval_application.py`

The project also evaluates the application as a user-facing system rather than treating RAG metrics as sufficient.

Three custom GEval metrics are used:

### Correctness

Reference-based evaluation of whether the answer is factually correct relative to the expected answer.

### Completeness

Evaluates whether the response covers the important points expected in the reference answer.

### Style

Evaluates teaching tone and communication style independently of correctness and completeness.

This separation is intentional.

For example:

```text
Correct but incomplete
```

should not receive the same diagnosis as:

```text
Complete but factually incorrect
```

GEval is specifically designed for custom, use-case-specific LLM-as-a-judge criteria, making it appropriate for these application-level dimensions.

---

# Safety and Behavioral Evaluation

The project has evolved beyond traditional RAG metrics into **behavioral and safety evaluation**.

These evaluations run against live outputs from the complete RAG pipeline.

---

## Toxicity Evaluation

### `evals/eval_toxicity.py`

Evaluates whether generated responses exhibit inappropriate or toxic behavior.

The project uses a custom GEval rather than treating a generic toxicity score as sufficient for the application's specific behavior.

---

# Prompt Leakage Evaluation

### `evals/eval_leakage.py`

Tests whether the assistant reveals:

* hidden system prompts
* internal instructions
* private operating rules
* protected implementation instructions

The evaluation distinguishes between:

```text
High-level capability explanation
```

and:

```text
Actual hidden-instruction disclosure
```

The goal is behavioral evaluation rather than simple keyword matching.

---

# Course-Content Leakage Evaluation

The system is also tested against attempts to extract protected course material.

The evaluation distinguishes legitimate educational behavior from excessive reproduction.

Allowed behavior includes:

* explaining concepts
* summarizing concepts
* teaching from retrieved knowledge

The system should avoid:

* dumping raw retrieved chunks
* reproducing substantial transcript content
* facilitating systematic extraction of the underlying corpus

This creates a dedicated evaluation dimension for **content protection** rather than conflating it with factual correctness.

---

# PII Leakage Evaluation

PII behavior is evaluated separately.

The project intentionally uses a custom GEval instead of relying solely on the built-in `PIILeakageMetric`.

The reason is behavioral:

A response such as:

```text
"I won't repeat your phone number."
```

should not automatically be treated as leaking a phone number merely because the response contains the phrase "phone number".

The custom evaluator instead asks whether the response actually discloses a specific sensitive value.

The distinction is:

```text
Mentioning a protected category
        ≠
Disclosing the protected value
```

This makes the evaluation more aligned with the application's actual safety requirement.

---

# Scope Adherence Evaluation

### `evals/eval_scope_safety.py`

The latest behavioral evaluation measures whether the assistant remains within its intended teaching-assistant scope.

The golden dataset contains three categories:

```text
ANSWER
DECLINE
PARTIAL
```

### ANSWER

The assistant should answer legitimate course-related questions.

### DECLINE

The assistant should avoid performing unrelated general-purpose tasks.

### PARTIAL

The assistant should:

1. answer the in-scope portion
2. avoid performing the unrelated portion

The evaluator also tests adversarial attempts such as:

* role switching
* jailbreak-style instructions
* requests to change the assistant's role

Importantly, the scope metric evaluates **scope behavior only** and does not intentionally reward or penalize:

* factual correctness
* completeness
* toxicity
* leakage
* style

This separation keeps the evaluation dimensions interpretable.

---

# Evaluation Matrix

| Layer              | Metric / Evaluation    | Purpose                                      |
| ------------------ | ---------------------- | -------------------------------------------- |
| Retrieval          | Contextual Recall      | Measures retrieval coverage                  |
| Retrieval          | Contextual Precision   | Measures retrieval focus                     |
| Retrieval          | Contextual Relevancy   | Measures usefulness of retrieved context     |
| Generation         | Faithfulness           | Measures grounding in context                |
| Generation         | Answer Relevancy       | Measures answer-query alignment              |
| Application        | Correctness            | Reference-based factual quality              |
| Application        | Completeness           | Reference-based coverage                     |
| Application        | Style                  | Teaching communication quality               |
| Safety             | Toxicity               | Detects inappropriate response behavior      |
| Security           | Prompt Leakage         | Protects hidden instructions                 |
| Content Protection | Course Content Leakage | Prevents excessive source reproduction       |
| Privacy            | PII Leakage            | Prevents disclosure of sensitive values      |
| Behavioral         | Scope Adherence        | Keeps the assistant within its intended role |

---

# Results

## Retriever Tuning

The retriever was iteratively tuned by changing:

* retrieval candidate pool
* embedding model
* chunk size

The resulting reranked retriever improved over the earlier configuration.

| Metric               |   Before |        After |
| -------------------- | -------: | -----------: |
| Contextual Recall    | 0.82 avg | **0.91 avg** |
| Recall Pass Rate     |      80% |      **93%** |
| Contextual Precision | 0.86 avg | **0.90 avg** |
| Precision Pass Rate  |      80% |      **87%** |

The final configuration uses:

```text
Embedding:
BAAI/bge-base-en-v1.5

Chunk size:
500

Chunk overlap:
100

Candidate pool:
20

Reranker:
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# Full Pipeline Results

Earlier recorded results from `eval_rag_pipeline.py` using the Claude Haiku judge:

| Metric               | Average Score |    Pass Rate |
| -------------------- | ------------: | -----------: |
| Contextual Relevancy |          0.76 |  80% (12/15) |
| Faithfulness         |     0.98–0.99 | 100% (15/15) |
| Answer Relevancy     |     0.92–0.95 |  87% (13/15) |

These results should be interpreted as **evaluation snapshots**, not permanent guarantees. LLM-as-a-judge scores can change with model versions, prompts, datasets, and evaluation configuration.

---

## Operational Metrics

Beyond correctness, the system measures real-world operational performance:

### Reliability

**`evals/eval_reliability.py`** — Measures whether the RAG application can serve requests without failing.

Tracks:
* Success rate
* Error rate
* Retry behavior (exponential backoff with 2 retries, 0.5s base delay)

**Result:**
* **100% success rate** (20/20 requests)
* 0% error rate
* 0% retry rate

---

### Latency

**`evals/eval_latency.py`** — Measures response time with percentile-based reporting (P50, P95, P99).

Reports both:
* **End-to-end latency** — total time from query to complete answer
* **Time-to-first-token (TTFT)** — perceived latency (how long until user sees first response token)

Component breakdown:
* Retrieval: ~240ms mean
* Generation: ~1820ms mean (dominant bottleneck)

**Latency Optimization:** 
Rewriting the generator prompt to favor conciseness over thoroughness reduced:
* Generation time: 3493ms → 1820ms mean (**48% improvement**)
* Answer length: 1090 → 449 characters (**59% reduction**)

**Result:**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| p95 end-to-end | ≤ 3000ms | **2698ms** | ✓ PASS |
| p95 TTFT | ≤ 1200ms | **1160ms** | ✓ PASS |

---

### Cost

**`evals/eval_cost.py`** — Measures token usage and projects cost at scale.

Tracks:
* Input tokens
* Output tokens
* Cached tokens (Anthropic prompt caching)
* Daily/monthly projections at configurable QPS

**Result:**
* **$0.000337 per query** (~0.032 INR per query)
* **$0.67 per day** at 2000 queries/day (~64 INR/day)
* Budget: ≤ $0.0015/query → ✓ PASS

---

# Claude as the Evaluation Judge

The project currently uses Claude for both:

```text
Generation
    +
Evaluation
```

The evaluation layer historically used a custom:

```python
DeepEvalBaseLLM
```

adapter backed by:

```text
Anthropic API
+
Instructor
+
Pydantic schemas
```

The adapter forces structured, schema-validated outputs from Claude for DeepEval metrics.

This keeps the evaluation pipeline on Anthropic credits and avoids requiring an OpenAI API key.

DeepEval now supports Anthropic models directly through `AnthropicModel`, as well as custom `DeepEvalBaseLLM` implementations. The custom adapter remains documented here because this project uses schema-controlled Claude responses and was built around its specific evaluation configuration.

---

# Model Compatibility Note

`claude-haiku-4-5` is currently used as the default evaluation judge.

A Sonnet model was experimentally tested for `ContextualRelevancyMetric`, but the project encountered a reproducible structured-output/tool-formatting failure in that configuration. After repeated retries, the output continued to follow an incompatible parameter structure.

The evaluation pipeline therefore remains pinned to the Haiku configuration that produced stable structured evaluation results.

This is documented as an implementation constraint rather than hiding the experiment.

---

# Golden Datasets

The project uses explicit golden datasets rather than relying exclusively on generated evaluation cases.

Current datasets include:

```text
goldens/
├── retriever_goldens.json
├── faithfulness_dataset.json
├── correctness_goldens.json
├── toxicity_goldens.json
├── leakage_goldens.json
└── scope_goldens.json
```

The datasets cover different evaluation dimensions.

### Retriever Goldens

Question + ideal retrieval answer/context.

### Faithfulness Dataset

Query + ideal context for isolated generator evaluation.

### Correctness Goldens

Question + expected answer for application-level evaluation.

### Toxicity Goldens

Inputs designed to exercise inappropriate-response behavior.

### Leakage Goldens

Inputs covering:

* prompt leakage
* course-content leakage
* PII leakage

### Scope Goldens

Inputs representing:

* ANSWER
* DECLINE
* PARTIAL

behavior.

---

# Golden Generation

### `goldens/generate_goldens.py`

DeepEval's `Synthesizer` is used to help generate additional candidate golden cases from transcript material.

Generated goldens are treated as **draft evaluation data** and should be manually reviewed before being promoted into the trusted evaluation set.

This avoids treating synthetic evaluation data as automatically authoritative.

---

# Chroma Inspection

### `export_chroma_chunks.py`

Exports stored Chroma chunks with:

* chunk ID
* text
* metadata

This is useful for manually inspecting retrieval behavior and constructing high-quality golden contexts.

It also makes retrieval debugging easier because the actual indexed corpus can be inspected independently from the RAG pipeline.

---

# Project Structure

```text
rag-eval-deepeval/
│
├── data/
│   └── *.vtt
│
├── src/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── evals/
│   ├── eval_retriever.py
│   ├── eval_retriever_with_reranker.py
│   ├── eval_generator.py
│   ├── eval_rag_pipeline.py
│   ├── eval_application.py
│   ├── eval_toxicity.py
│   ├── eval_leakage.py
│   └── eval_scope_safety.py
│
├── goldens/
│   ├── retriever_goldens.json
│   ├── faithfulness_dataset.json
│   ├── correctness_goldens.json
│   ├── toxicity_goldens.json
│   ├── leakage_goldens.json
│   ├── scope_goldens.json
│   └── generate_goldens.py
│
├── resources/
│   └── deepeval_intro.py
│
├── export_chroma_chunks.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Setup

The project uses `uv` for environment and dependency management.

```bash
uv sync
```

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your-key-here
```

The project does not require an OpenAI API key for its current Claude-based configuration.

---

# Running the Pipeline

## Run the retriever

```bash
python src/retriever.py
```

## Run the generator

```bash
python src/generator.py
```

## Run the complete RAG pipeline

```bash
python -m src.rag_pipeline
```

---

# Running Evaluations

## Retriever

```bash
python -m evals.eval_retriever
```

## Retriever + reranker

```bash
python -m evals.eval_retriever_with_reranker
```

## Generator

```bash
python -m evals.eval_generator
```

## Full RAG pipeline

```bash
python -m evals.eval_rag_pipeline
```

## Application quality

```bash
python -m evals.eval_application
```

## Toxicity

```bash
python -m evals.eval_toxicity
```

## Leakage

```bash
python -m evals.eval_leakage
```

## Scope adherence

```bash
python -m evals.eval_scope_safety
```

---

# Generate Additional Golden Cases

```bash
python goldens/generate_goldens.py
```

Review generated cases before using them as trusted evaluation data.

---

# Inspect Chroma Chunks

```bash
python export_chroma_chunks.py
```

---

# Interactive Application

Launch the Streamlit interface with:

```bash
streamlit run <streamlit-entrypoint>
```

The UI provides an interactive surface for testing the live RAG system.

---

# Design Principles

## 1. Evaluate components independently

A final answer score alone cannot tell whether a failure came from retrieval or generation.

Therefore:

```text
Retriever
   ↓
Generator
   ↓
Full Pipeline
```

are evaluated separately.

---

## 2. Separate quality from safety

Correctness and safety are different dimensions.

A response can be:

```text
Correct but unsafe
```

or:

```text
Safe but incorrect
```

The evaluation suite therefore keeps these dimensions separate.

---

## 3. Use custom metrics where generic metrics are insufficient

Standard RAG metrics are useful for retrieval and generation quality.

They are not sufficient for application-specific requirements such as:

* scope adherence
* prompt protection
* course-content protection
* application-specific correctness
* teaching style

GEval is therefore used for custom behavioral criteria.

---

## 4. Evaluate live pipeline behavior

Safety and application-level evaluations execute the actual pipeline rather than evaluating handcrafted responses in isolation.

```text
Golden Input
     ↓
Live RAG Pipeline
     ↓
Actual Output
     ↓
LLM Judge
     ↓
Metric Score
```

This makes the tests closer to real application behavior.

---

## 5. Treat evaluation datasets as engineering artifacts

Golden datasets are versioned alongside the implementation.

Changes to:

* prompts
* retrieval
* reranking
* models
* safety rules

can therefore be evaluated against a stable test set.

---

# Known Limitations

This project is an evaluation-focused RAG system, not a claim of production-grade safety certification.

Important limitations include:

* LLM-as-a-judge scores are model-dependent.
* Evaluation results can vary across judge-model versions.
* Golden datasets are relatively small.
* Synthetic goldens require human review.
* The safety evaluations are behavioral tests, not formal security guarantees.
* Cross-encoder reranking adds latency compared with pure vector retrieval.
* Local embedding inference requires additional compute.
* Chroma persistence is local rather than a distributed production vector database.
* Evaluation runs consume Anthropic API credits.
* The current evaluation suite does not constitute a statistically exhaustive benchmark.

These limitations are intentionally documented rather than hidden behind aggregate scores.

---

# Why This Project Exists

Many RAG implementations stop at:

```text
Retrieve → Generate
```

and then demonstrate a few example questions.

This project focuses on the harder question:

> **How do you know whether the RAG application is actually behaving correctly?**

The system therefore evaluates the pipeline at multiple levels:

```text
Retrieval quality
       +
Generation quality
       +
Application quality
       +
Safety
       +
Security
       +
Behavioral adherence
```

The goal is not to produce one impressive number.

The goal is to make RAG failures **observable, reproducible, and diagnosable**.

---

# Evaluation Philosophy

The central principle is:

> **A RAG system should be evaluated as a system, not just as a language model.**

A useful evaluation stack therefore looks like:

```text
                    ┌──────────────────────┐
                    │      RAG System      │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        Retrieval          Generation       Application
             │                 │                 │
             ▼                 ▼                 ▼
       Recall /           Faithfulness      Correctness
       Precision          Relevancy         Completeness
       Relevancy                             Style
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                       Safety / Security
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
          Toxicity          Leakage           Scope
```

This layered approach makes it possible to answer not only:

> "Did the system fail?"

but:

> **"Which layer failed, why did it fail, and can we measure whether a change actually improved it?"**

---

# Current Status

**Implemented:**

* [x] Transcript ingestion
* [x] Local BGE embeddings
* [x] Persistent Chroma retrieval
* [x] Cross-encoder reranking
* [x] Claude-based generation
* [x] Faithfulness-first generation policy
* [x] Retriever evaluation
* [x] Reranker evaluation
* [x] Generator evaluation
* [x] End-to-end RAG evaluation
* [x] Application-level evaluation
* [x] Correctness evaluation
* [x] Completeness evaluation
* [x] Style evaluation
* [x] Toxicity evaluation
* [x] Prompt leakage evaluation
* [x] Course-content leakage evaluation
* [x] PII leakage evaluation
* [x] Scope adherence evaluation
* [x] Golden datasets
* [x] Golden generation workflow
* [x] Chroma inspection/export tooling
* [x] Interactive Streamlit interface

---

# Technology Stack

| Layer                   | Technology                           |
| ----------------------- | ------------------------------------ |
| Language                | Python                               |
| Package Management      | uv                                   |
| RAG Framework           | LangChain                            |
| Embeddings              | BAAI/bge-base-en-v1.5                |
| Vector Store            | Chroma                               |
| Reranker                | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation              | Claude                               |
| Evaluation              | DeepEval                             |
| Custom Evaluation       | GEval                                |
| Structured Judge Output | Instructor + Pydantic                |
| UI                      | Streamlit                            |
| Source Data             | YouTube `.vtt` transcripts           |

---

# Further Work

Completed:
* ✓ Reliability measurement (`eval_reliability.py`)
* ✓ Latency benchmarking (`eval_latency.py`) — including TTFT and component decomposition
* ✓ Cost-per-query measurement (`eval_cost.py`)
* ✓ Toxicity evaluation with custom GEval
* ✓ Leakage evaluation (prompt, content, PII) with custom GEval
* ✓ Scope safety evaluation

Potential future improvements include:

* larger and more diverse evaluation datasets
* automated regression testing in CI
* evaluation-result persistence and comparison across runs
* model comparison experiments
* embedding-model comparison
* reranker comparison
* adversarial evaluation expansion
* automated evaluation dashboards
* statistical confidence intervals over evaluation results
* trace-level observability for individual RAG failures

---

## Summary

`rag-eval-deepeval` is a complete evaluation-oriented RAG system designed around a simple principle:

```text
Build the RAG system.
        ↓
Measure each component.
        ↓
Measure the complete application.
        ↓
Test safety and behavioral boundaries.
        ↓
Use the results to drive engineering changes.
```

Rather than treating RAG evaluation as a final benchmark, the project treats evaluation as part of the **development loop itself**.

**Retrieval → Reranking → Generation → Evaluation → Diagnosis → Iteration**
