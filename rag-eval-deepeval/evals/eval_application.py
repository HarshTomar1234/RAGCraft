import json
import instructor
from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
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


JUDGE_MODEL = ClaudeJudge()


# 1. LOAD queries + ideal answers
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["question"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],   # what the generator produced
            expected_output=g["ideal_answer"],  # the CORRECT reference answer
        )
    )


# 3. THREE APPLICATION-LEVEL QUALITY METRICS

# 3a. CORRECTNESS — reference-based, judges TRUTH (not coverage or length)
# You can also give criteria and then geval will generate a evaluation steps for you using Chain of Thought (CoT) reasoning.
# The rubric is a list of score ranges and expected outcomes for each range.
# You can also provide evaluation steps earlier skipping criteria since geval will generate them for you and in multiple iteration there will be some variation in the evaluation steps.
# so providing evaluation steps prevent that variation and make the evaluation more consistent.

# So the right approach is to use criteria as parameter in earlier testing but after some iteration you can provide evaluation steps to make the evaluation more consistent and avoid variation in the evaluation steps.
# same goes for rubric you can not provide rubric as parameter in earlier testing  and let model decide the way to score but after some iteration you can provide rubric to make the evaluation more consistent and avoid variation in the rubric/scoring sense.
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points — omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
        Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# 3b. COMPLETENESS — reference-based, judges COVERAGE (not correctness)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# 3c. STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
        Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
        Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
        Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)



def run(rag):
    """Evaluate application-level quality (correctness, completeness, style)."""
    # 1. LOAD queries + ideal answers
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["question"])          # retrieve → rerank → generate

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],   # what the generator produced
                expected_output=g["ideal_answer"],  # the CORRECT reference answer
            )
        )

    # 3. THREE APPLICATION-LEVEL QUALITY METRICS

    # 3a. CORRECTNESS — reference-based, judges TRUTH (not coverage or length)
    # You can also give criteria and then geval will generate a evaluation steps for you using Chain of Thought (CoT) reasoning.
    # The rubric is a list of score ranges and expected outcomes for each range.
    # You can also provide evaluation steps earlier skipping criteria since geval will generate them for you and in multiple iteration there will be some variation in the evaluation steps.
    # so providing evaluation steps prevent that variation and make the evaluation more consistent.

    # So the right approach is to use criteria as parameter in earlier testing but after some iteration you can provide evaluation steps to make the evaluation more consistent and avoid variation in the evaluation steps.
    # same goes for rubric you can not provide rubric as parameter in earlier testing  and let model decide the way to score but after some iteration you can provide rubric to make the evaluation more consistent and avoid variation in the rubric/scoring sense.
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points — omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 3b. COMPLETENESS — reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 3c. STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
            "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
            "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
            "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
            "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 4. EVALUATE — all three together
    result = evaluate(test_cases=test_cases, metrics=[correctness, completeness, style])
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())