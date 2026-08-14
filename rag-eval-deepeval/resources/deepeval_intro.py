import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from pydantic import BaseModel

load_dotenv()


# AnswerRelevancyMetric needs a judge LLM; deepeval defaults to OpenAI.
# This routes it through Claude instead. instructor forces schema-validated
# JSON out of Anthropic's plain messages API, which the metric requires.
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

# --- Test case 1: a good answer (should PASS) ---
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

# --- Test case 2: an off-topic answer (should FAIL) ---
case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)

# --- One metric, judged by an LLM (pinned for reproducibility) ---
metric = AnswerRelevancyMetric(threshold=0.7, model=ClaudeJudge(), include_reason=True)

# --- Run BOTH cases through the metric, with a printed report ---
evaluate(test_cases=[case_1, case_2], metrics=[metric])