import json
from src.llm import generate

SYNTHESIZER_SYSTEM_PROMPT = """You are synthesizing a final answer from a chain of resolved sub-questions.

Given the original question and a sequence of resolved sub-questions with their answers, 
produce a final answer to the original question.

Also assess your confidence: if any step's answer was "Could not find sufficient evidence" 
or seems uncertain/guessed, mark confidence as "low". Otherwise mark it "high".

Respond ONLY with a JSON object in this exact format:
{"answer": "the final answer", "confidence": "high", "reasoning": "brief explanation of how the steps led to this answer"}
"""


def synthesize(original_question: str, steps: list[dict], model: str = "llama3.1") -> dict:
    steps_text = "\n".join(
        f"{i+1}. {s['contextualized_question']} -> {s['answer']}"
        for i, s in enumerate(steps)
    )
    prompt = f"Original question: {original_question}\n\nResolved steps:\n{steps_text}"

    raw = generate(prompt, model=model, system=SYNTHESIZER_SYSTEM_PROMPT)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Warning: synthesizer output not valid JSON:\n{raw}")
        return {
            "answer": steps[-1]["answer"] if steps else "Unable to determine.",
            "confidence": "low",
            "reasoning": "Synthesizer output could not be parsed; falling back to last step's answer.",
        }


if __name__ == "__main__":
    from src.retriever import Retriever
    from src.agent_loop import run_agent

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

    original_question = "What is the population of the country whose capital contains the Eiffel Tower?"
    result = run_agent(original_question, sub_questions, retriever)
    final = synthesize(original_question, result["steps"])

    print(f"Question: {original_question}\n")
    print(f"Final Answer: {final['answer']}")
    print(f"Confidence: {final['confidence']}")
    print(f"Reasoning: {final['reasoning']}")