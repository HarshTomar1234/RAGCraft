"""
src/generator.py — the GENERATOR component.

Given a query and context (retrieved chunks), produce an answer grounded in
the context. The prompt is faithfulness-first: answer ONLY from the context,
and abstain when the context doesn't contain the answer.

    from src.generator import generate
    answer = generate("what is drift?", ["chunk text 1", "chunk text 2"])
"""

# from langchain_openai import ChatOpenAI  
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# faithfulness-first prompt: ground every claim in the context, abstain if unsure
prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for a course on LLM evaluations.
Answer the student's question using ONLY the information in the context provided below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- Answer thoroughly: identify every distinct part of the question and address each
  one, using every relevant point the context provides.
- Write in flowing, conversational prose, the way a teacher explains something out
  loud — not as a bulleted or numbered list. Only use a list when the question
  genuinely calls for enumeration.
- Explain the intuition first in plain language, and briefly unpack any technical
  term you use.
- Keep the answer only as long as the question needs. Do not add unrelated
  information, restate points, or build up more context than the question asked for.
- If the context only partially covers the question, answer the part it covers and
  say plainly what's missing — don't discard a partial answer.
- If the context does not contain enough information to answer at all, say:
  "I don't have enough information in the course material to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


# quick manual test: python src/generator.py
if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]
    print(generate("what is online eval?", ctx))