import json
import os

# raise deepeval's outer per-test-case budget (default 180s) before it's read;
# 3 metrics x several sequential Haiku calls each needs more room than that
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
# JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


# Metrics need a judge LLM; deepeval defaults to OpenAI.
# This routes it through Claude instead. instructor forces schema-validated
# JSON out of Anthropic's plain messages API, which the metrics require.
class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.model_name = model
        self.client = instructor.from_anthropic(Anthropic())

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


JUDGE_MODEL = ClaudeJudge()  # sonnet-5 deterministically malforms ContextualRelevancyMetric's tool call on this dataset; haiku doesn't


# 1. LOAD queries (we only need the queries — context comes from the pipeline now)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],       # what the generator produced
            retrieval_context=result["context"],  # what the RETRIEVER returned
        )
    )


# 3. THE THREE TRIAD METRICS
metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE
# parallel, but capped: unbounded concurrency floods the rate limit and trips
# deepeval's per-test-case timeout
evaluate(test_cases=test_cases, metrics=metrics, async_config=AsyncConfig(max_concurrent=5))