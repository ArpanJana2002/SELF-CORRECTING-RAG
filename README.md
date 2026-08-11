# Self-Correcting RAG Agent (LangGraph)

A fully functional Corrective-RAG (CRAG) style agent: it retrieves, **grades its own
retrieval quality**, falls back to live web search when the knowledge base falls short,
and **grades its own generated answer** for hallucination and relevance before returning
it — looping and correcting itself instead of blindly returning the first draft.

## Architecture

```
retrieve → grade_documents → [enough relevant docs?]
                                 ├─ yes → generate
                                 └─ no  → transform_query → web_search → generate

generate → [grade_generation]
              ├─ grounded + answers question → END
              ├─ not grounded (hallucinated)  → generate again
              └─ grounded but off-topic       → transform_query (retry retrieval)
```

Bounded by `MAX_CORRECTION_LOOPS` so it can never loop forever.

| Stage | What it does |
|---|---|
| `retrieve` | Vector-similarity search against Chroma |
| `grade_documents` | An LLM call grades **each chunk** relevant/irrelevant and drops the irrelevant ones |
| `transform_query` | LLM rewrites the question for better retrieval/search |
| `web_search` | Tavily web search fills the gap when local docs aren't enough (the "corrective" step) |
| `generate` | LLM answers using only the surviving context |
| `grade_generation` | Two more LLM checks: (1) is the answer grounded in the context (no hallucination)? (2) does it actually answer the question? |

## Tech stack

- **LangGraph** — the stateful graph/orchestration engine (cycles + conditional routing)
- **LangChain** — prompts, document loaders, text splitting, retriever interface
- **Chroma** — local, persistent vector store
- **Embeddings** — swappable: free local `sentence-transformers` (CPU, no API key) or OpenAI
- **LLM** — swappable: Anthropic Claude or OpenAI GPT, via `with_structured_output` for reliable grading
- **Tavily** — web search API used as the corrective fallback
- **Pydantic** — structured/typed grader outputs (`binary_score: yes/no`)

You can swap any of these (e.g. FAISS instead of Chroma, Ollama for a fully local LLM,
Bing/SerpAPI instead of Tavily) — each integration point is isolated in `src/config.py`,
`src/ingestion.py`, and `src/nodes.py`.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and TAVILY_API_KEY
```

Minimum required keys:
- One LLM key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`)
- `TAVILY_API_KEY` — free tier at https://tavily.com (only used when correction kicks in)
- No key needed for embeddings if you leave `EMBEDDINGS_PROVIDER=local` (default)

## Usage

```bash
# 1. Ingest your documents (drop .txt/.md files into ./data, or point --path elsewhere)
python main.py ingest --path ./data

# 2. Ask a question
python main.py ask "What is corrective RAG and how does it differ from plain RAG?"

# 3. Or chat interactively
python main.py chat
```

You can also ingest live web pages instead of/in addition to local files:
```bash
python main.py ingest --url https://example.com/docs/page1 --url https://example.com/docs/page2
```

### Programmatic use

```python
from src.graph import run

result = run("How does LangGraph implement cycles?")
print(result["generation"])
print("used web search:", result["used_web_search"])
```

## Project layout

```
self_correcting_rag/
├── main.py                # CLI (ingest / ask / chat)
├── requirements.txt
├── .env.example
├── data/                  # sample docs to ingest
├── vectorstore/           # Chroma persistence (created on first ingest)
└── src/
    ├── config.py          # env-driven LLM/embeddings/vectorstore config
    ├── state.py            # LangGraph shared state (TypedDict)
    ├── ingestion.py        # load → split → embed → persist to Chroma
    ├── graders.py          # structured LLM graders + query rewriter
    ├── nodes.py             # graph node functions + conditional routers
    └── graph.py            # StateGraph wiring (the actual LangGraph graph)
```
<img width="943" height="130" alt="Screenshot 2026-08-11 120101" src="https://github.com/user-attachments/assets/7ddf78d5-f9ee-4c2c-b794-2da9d6f2d3fb" />

## Extending it

- **Swap the vector store**: replace the Chroma calls in `src/ingestion.py` with FAISS,
  Pinecone, Weaviate, etc. — the rest of the graph doesn't care.
- **Add a re-ranker**: drop a cross-encoder reranking step between `retrieve` and
  `grade_documents` for higher-precision filtering before the LLM grader runs.
- **Self-RAG style adaptive retrieval**: add a router node before `retrieve` that first
  asks the LLM whether retrieval is even needed for this question.
- **Human-in-the-loop**: LangGraph supports checkpointing — add a `MemorySaver` /
  `interrupt_before` on `generate` to have a human approve corrected answers.
- **Streaming**: swap `app.invoke` for `app.astream_events` in a web backend (e.g. FastAPI)
  to stream node-by-node progress to a frontend.


