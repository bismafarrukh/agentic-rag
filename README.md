## Architecture

Traditional RAG retrieves once and generates an answer. That breaks down on questions that require chaining multiple facts together (multi-hop questions), since no single retrieval contains the full answer.

This project uses an **agentic** approach: instead of a fixed retrieve-then-generate pipeline, an LLM-driven agent decides *when* to retrieve, *what* to search for next, and *when* it has enough evidence to answer.

```
User Query
   │
   ▼
Query Planner (LLM)
   └─ decomposes the question into sub-questions
   │
   ▼
Retrieval Loop (repeated per sub-question)
   ├─ Retrieve  → vector search over the corpus
   ├─ Rerank    → (optional) reorder results by relevance
   ├─ Judge     → "do I have enough evidence to answer this sub-question?"
   └─ If not enough → generate a follow-up query → retrieve again
   │
   ▼
Answer Synthesizer (LLM)
   └─ combines all retrieved evidence + reasoning trace into a final answer
   │
   ▼
Faithfulness Checker
   └─ flags any claims in the answer not supported by retrieved text
```

### Component overview

| Component | File | Role |
|---|---|---|
| Retriever | `src/retriever.py` | Embeds documents/queries, stores and searches vectors |
| Query Planner | `src/planner.py` | Breaks a complex question into sub-questions |
| Agent Loop | `src/agent_loop.py` | Controls the multi-hop retrieve → judge → re-retrieve cycle |
| Synthesizer | `src/synthesizer.py` | Generates the final answer from all gathered evidence |
| Evaluator | `src/evaluator.py` | Measures retrieval recall and answer faithfulness against a benchmark |

### Why "agentic" matters here
A plain RAG pipeline treats retrieval as a single fixed step. An agentic pipeline treats retrieval as a **tool** the LLM can choose to call, multiple times, adapting its next query based on what it just learned — closer to how a person researches an answer by following one clue to the next, rather than looking something up once and stopping.

Each component above is intentionally kept independent (plain inputs/outputs, no shared internal state) so pieces can be swapped or tested individually — e.g. swapping Qdrant for another vector store only touches `retriever.py`.