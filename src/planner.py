import json
from src.llm import generate

PLANNER_SYSTEM_PROMPT = """You are a query decomposition assistant for a multi-hop question answering system.

Given a complex question, break it into a minimal ordered list of simple sub-questions that, 
answered in sequence, would let someone answer the original question.

If the question is already simple and answerable with a single retrieval, return just one sub-question 
(the original question itself).

Respond ONLY with a JSON array of strings, nothing else. No explanation, no markdown formatting.

Example:
Question: "What is the population of the country whose capital contains the Eiffel Tower?"
Response: ["Which country's capital contains the Eiffel Tower?", "What is the population of that country?"]
"""


def plan(question: str, model: str = "llama3.1") -> list[str]:
    raw = generate(question, model=model, system=PLANNER_SYSTEM_PROMPT)

    # Models sometimes wrap JSON in markdown fences despite instructions — strip if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        sub_questions = json.loads(cleaned)
        if not isinstance(sub_questions, list):
            raise ValueError("Expected a JSON list")
        return sub_questions
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: failed to parse planner output as JSON. Raw output:\n{raw}")
        # Fallback: treat the whole question as a single sub-question
        return [question]


if __name__ == "__main__":
    question = "What is the population of the country whose capital contains the Eiffel Tower?"
    sub_questions = plan(question)
    print("Sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"{i}. {sq}")