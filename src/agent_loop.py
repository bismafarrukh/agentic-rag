import json
from src.llm import generate
from src.retriever import Retriever


REWRITE_SYSTEM_PROMPT = """You rewrite a sub-question into a standalone question using known facts.

Given prior resolved facts and a sub-question that may contain references like "that country" 
or "this city", rewrite it as a self-contained question with no ambiguous references.

If the sub-question is already standalone, return it unchanged.

Respond ONLY with the rewritten question as plain text. No explanation, no quotes.
"""

JUDGE_SYSTEM_PROMPT = """You are evaluating whether retrieved evidence is sufficient to answer a question.

Given a question and retrieved passages, respond ONLY with a JSON object in this exact format:
{"sufficient": true, "answer": "the answer", "refined_query": null}
or
{"sufficient": false, "answer": null, "refined_query": "a better search query to try instead"}

Base the answer ONLY on the provided passages. If the passages don't contain the answer, 
set sufficient to false and suggest a refined_query that might retrieve better evidence.
"""


def rewrite_with_context(sub_question: str, resolved: list[dict], model: str = "llama3.1") -> str:
    if not resolved:
        return sub_question

    facts = "\n".join(f"- {r['question']} -> {r['answer']}" for r in resolved)
    prompt = f"Known facts so far:\n{facts}\n\nSub-question to rewrite:\n{sub_question}"
    return generate(prompt, model=model, system=REWRITE_SYSTEM_PROMPT).strip()


def judge(question: str, passages: list[dict], model: str = "llama3.1") -> dict:
    passages_text = "\n".join(f"- {p['text']}" for p in passages)
    prompt = f"Question: {question}\n\nRetrieved passages:\n{passages_text}"
    raw = generate(prompt, model=model, system=JUDGE_SYSTEM_PROMPT)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Warning: judge output not valid JSON:\n{raw}")
        return {"sufficient": False, "answer": None, "refined_query": question}


def answer_sub_question(question: str, retriever: Retriever, max_retries: int = 2, model: str = "llama3.1") -> dict:
    query = question
    result = None

    for attempt in range(max_retries):
        passages = retriever.search(query)
        result = judge(question, passages, model=model)

        if result.get("sufficient"):
            return {"answer": result["answer"], "evidence": passages}

        refined = result.get("refined_query")
        if not refined or refined == query:
            break
        query = refined

    # Ran out of retries — return best-effort, but mark it as unresolved
    return {"answer": result.get("answer") or "Could not find sufficient evidence.", "evidence": passages}


def run_agent(original_question: str, sub_questions: list[str], retriever: Retriever, model: str = "llama3.1") -> dict:
    resolved = []

    for sub_q in sub_questions:
        contextualized_q = rewrite_with_context(sub_q, resolved, model=model)
        result = answer_sub_question(contextualized_q, retriever, model=model)
        resolved.append({
            "question": sub_q,
            "contextualized_question": contextualized_q,
            "answer": result["answer"],
            "evidence": result["evidence"],
        })

    return {"original_question": original_question, "steps": resolved}


if __name__ == "__main__":
    retriever = Retriever()
    retriever.add_documents([
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France's population is about 68 million.",
    ])

    sub_questions = [
        "Which city contains the Eiffel Tower?",
        "Which country is this city the capital of?",
        "What is the population of this country?",
    ]

    result = run_agent(
        "What is the population of the country whose capital contains the Eiffel Tower?",
        sub_questions,
        retriever,
    )

    for i, step in enumerate(result["steps"], 1):
        print(f"Step {i}: {step['contextualized_question']}")
        print(f"  Answer: {step['answer']}\n")