import json
from src.retriever import Retriever
from src.planner import plan
from src.agent_loop import run_agent
from src.synthesizer import synthesize


def retrieval_recall(question: str, gold_fact: str, retriever: Retriever, top_k: int = 5) -> bool:
    """Did the retriever return the gold fact within its top_k results for this question?"""
    results = retriever.search(question, top_k=top_k)
    return any(gold_fact.lower() in r["text"].lower() for r in results)


def answer_correct(predicted: str, gold_answer: str) -> bool:
    """Loose match: does the predicted answer contain the gold answer (case-insensitive)?"""
    return gold_answer.lower() in predicted.lower()


def evaluate(benchmark_path: str, retriever: Retriever, model: str = "llama3.1") -> dict:
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)

    results = []
    recall_hits = 0
    correct_hits = 0

    for item in benchmark:
        question = item["question"]
        gold_answer = item["gold_answer"]
        gold_fact = item["gold_fact"]

        # Retrieval recall — check directly against the original question
        recall_hit = retrieval_recall(question, gold_fact, retriever)
        recall_hits += int(recall_hit)

        # Full pipeline run
        sub_questions = plan(question, model=model)
        agent_result = run_agent(question, sub_questions, retriever, model=model)
        final = synthesize(question, agent_result["steps"], model=model)

        correct = answer_correct(final["answer"], gold_answer)
        correct_hits += int(correct)

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": final["answer"],
            "confidence": final["confidence"],
            "retrieval_recall_hit": recall_hit,
            "answer_correct": correct,
        })

    n = len(benchmark)
    summary = {
        "retrieval_recall": recall_hits / n if n else 0,
        "answer_accuracy": correct_hits / n if n else 0,
        "num_questions": n,
    }

    return {"summary": summary, "details": results}


if __name__ == "__main__":
    retriever = Retriever()
    retriever.add_documents([
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France's population is about 68 million.",
        "France is located in Europe.",
    ])

    report = evaluate("eval/benchmark_questions.json", retriever)

    print("=== Summary ===")
    print(f"Retrieval Recall: {report['summary']['retrieval_recall']:.2f}")
    print(f"Answer Accuracy: {report['summary']['answer_accuracy']:.2f}")
    print(f"Questions evaluated: {report['summary']['num_questions']}\n")

    print("=== Details ===")
    for d in report["details"]:
        print(f"Q: {d['question']}")
        print(f"  Gold: {d['gold_answer']} | Predicted: {d['predicted_answer']} | Correct: {d['answer_correct']}")
        print(f"  Retrieval hit: {d['retrieval_recall_hit']} | Confidence: {d['confidence']}\n")