"""
Node functions for the self-correcting RAG graph.

Flow (see src/graph.py for wiring):

  retrieve -> grade_documents -> [decide_to_generate]
      -> (relevant docs exist)      -> generate -> [grade_generation] -> END | transform_query
      -> (no/insufficient docs)     -> transform_query -> web_search -> generate -> ...
"""
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.state import GraphState
from src.ingestion import get_retriever
from src.graders import (
    get_document_grader,
    get_hallucination_grader,
    get_answer_grader,
    get_question_rewriter,
)

# --- lazily-initialized singletons (avoid re-building chains on every call) ---
_retriever = None
_doc_grader = None
_hallucination_grader = None
_answer_grader = None
_question_rewriter = None
_rag_chain = None
_web_search_tool = None


def _retriever_singleton():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def _rag_chain_singleton():
    global _rag_chain
    if _rag_chain is None:
        llm = config.get_llm(temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. Use the following "
                    "retrieved context to answer the question. If the context doesn't contain "
                    "the answer, say you don't know — do not make anything up. Keep the answer "
                    "concise (max ~4 sentences) and cite facts only from the given context.",
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        _rag_chain = prompt | llm
    return _rag_chain


def _web_search_tool_singleton():
    global _web_search_tool
    if _web_search_tool is None:
        if not config.TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Web-search-based correction requires a Tavily key "
                "(free tier at https://tavily.com)."
            )
        _web_search_tool = TavilySearchResults(k=3, tavily_api_key=config.TAVILY_API_KEY)
    return _web_search_tool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from the vector store for the current question."""
    print(f"---RETRIEVE--- (question: {state['question']!r})")
    docs = _retriever_singleton().invoke(state["question"])
    doc_texts = [d.page_content for d in docs]
    return {**state, "documents": doc_texts}


def grade_documents(state: GraphState) -> GraphState:
    """Filter retrieved docs, keeping only those graded relevant. Flag web search if needed."""
    print("---GRADE DOCUMENTS---")
    grader = get_document_grader()
    question = state["question"]
    filtered = []
    for d in state["documents"]:
        try:
            result = grader.invoke({"question": question, "document": d})
            score = result.binary_score.strip().lower()
        except Exception as e:
            print(f"  [grader error, keeping doc defensively] {e}")
            score = "yes"

        if score == "yes":
            print("  -> relevant")
            filtered.append(d)
        else:
            print("  -> NOT relevant, dropping")

    # If fewer than half the retrieved docs (or none) survived, trigger web search
    web_search_needed = "Yes" if len(filtered) == 0 else "No"
    return {**state, "documents": filtered, "web_search_needed": web_search_needed}


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the question to improve retrieval / web search quality."""
    print("---TRANSFORM QUERY---")
    rewriter = get_question_rewriter()
    result = rewriter.invoke({"question": state["question"]})
    new_question = result.content.strip()
    print(f"  rewritten question: {new_question!r}")
    return {**state, "question": new_question, "loop_count": state.get("loop_count", 0) + 1}


def web_search(state: GraphState) -> GraphState:
    """Corrective step: pull in live web results and append them as context."""
    print("---WEB SEARCH (corrective fallback)---")
    tool = _web_search_tool_singleton()
    results = tool.invoke({"query": state["question"]})
    web_texts = [r["content"] for r in results if isinstance(r, dict) and "content" in r]
    combined = state.get("documents", []) + web_texts
    return {**state, "documents": combined, "used_web_search": True}


def generate(state: GraphState) -> GraphState:
    """Generate an answer grounded in the current set of documents."""
    print("---GENERATE---")
    chain = _rag_chain_singleton()
    context = "\n\n---\n\n".join(state["documents"]) if state["documents"] else "(no context found)"
    result = chain.invoke({"question": state["original_question"], "context": context})
    return {**state, "generation": result.content}


# ---------------------------------------------------------------------------
# Conditional edges (routers)
# ---------------------------------------------------------------------------

def decide_to_generate(state: GraphState) -> str:
    """After grading docs: go straight to generate, or correct via query rewrite + web search."""
    print("---DECIDE: generate directly, or correct first?---")
    if state["web_search_needed"] == "Yes":
        print("  -> insufficient relevant docs, routing to correction path (transform_query)")
        return "transform_query"
    print("  -> sufficient relevant docs, routing to generate")
    return "generate"


def grade_generation(state: GraphState) -> str:
    """
    After generate: check
      1) is the answer grounded in the retrieved documents (no hallucination)?
      2) does the answer actually address the question?
    Loops back to transform_query/generate if not, bounded by MAX_CORRECTION_LOOPS.
    """
    print("---GRADE GENERATION (hallucination + answer quality)---")
    loop_count = state.get("loop_count", 0)
    if loop_count >= config.MAX_CORRECTION_LOOPS:
        print(f"  -> hit MAX_CORRECTION_LOOPS ({config.MAX_CORRECTION_LOOPS}), stopping here")
        return "useful"

    docs_text = "\n\n".join(state["documents"]) if state["documents"] else "(no context)"

    hallucination_grader = get_hallucination_grader()
    h_result = hallucination_grader.invoke(
        {"documents": docs_text, "generation": state["generation"]}
    )
    if h_result.binary_score.strip().lower() != "yes":
        print("  -> generation NOT grounded in documents -> regenerate")
        return "not supported"

    print("  -> generation is grounded in documents")
    answer_grader = get_answer_grader()
    a_result = answer_grader.invoke(
        {"question": state["original_question"], "generation": state["generation"]}
    )
    if a_result.binary_score.strip().lower() == "yes":
        print("  -> generation addresses the question -> DONE")
        return "useful"

    print("  -> generation does NOT address the question -> rewrite query and retry")
    return "not useful"
