import ollama


def generate(prompt: str, model: str = "llama3.1", system: str = None) -> str:
    """
    Sends a prompt to a local Ollama model and returns the text response.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


if __name__ == "__main__":
    answer = generate("What is the capital of France? Answer in one word.")
    print(answer)