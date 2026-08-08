import argparse
from src.retriever import Retriever
from src.planner import plan
from src.agent_loop import run_agent
from src.synthesizer import synthesize


def build_demo_corpus(retriever: Retriever):
    """Temporary demo corpus. Replace with real document loading once you have a real corpus."""
    retriever.add_documents([
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France's population is about 68 million.",
        "France is located in Europe.",
    ])


def answer_question(question: str, retriever: Retriever, model: str = "llama3.1", verbose: bool = False) -> dict:
    sub_questions = plan(question, model=model)
    agent_result = run_agent(question, sub_questions, retriever, model=model)
    final = synthesize(question, agent_result["steps"], model=model)

    if verbose:
        print("\n--- Sub-questions ---")
        for i, sq in enumerate(sub_questions, 1):
            print(f"{i}. {sq}")

        print("\n--- Reasoning steps ---")
        for i, step in enumerate(agent_result["steps"], 1):
            print(f"{i}. {step['contextualized_question']} -> {step['answer']}")

    return final


def main():
    parser = argparse.ArgumentParser(description="Agentic multi-hop RAG")
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--model", type=str, default="llama3.1", help="Ollama model to use")
    parser.add_argument("--verbose", action="store_true", help="Show sub-questions and reasoning steps")
    args = parser.parse_args()

    retriever = Retriever()
    build_demo_corpus(retriever)

    final = answer_question(args.question, retriever, model=args.model, verbose=args.verbose)

    print("\n--- Final Answer ---")
    print(f"Answer: {final['answer']}")
    print(f"Confidence: {final['confidence']}")
    print(f"Reasoning: {final['reasoning']}")


if __name__ == "__main__":
    main()