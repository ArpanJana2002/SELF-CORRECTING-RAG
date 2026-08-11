"""
Shared state that flows through every node of the LangGraph graph.
"""
from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Attributes:
        question:        the current user question (may be rewritten mid-run)
        original_question: the question exactly as the user asked it
        generation:       the LLM's generated answer (filled in by `generate`)
        documents:        list of retrieved/relevant document texts (as strings)
        web_search_needed: "Yes" / "No" flag set by the document grader
        loop_count:       how many correction loops we've run (safety valve)
        used_web_search:  whether web search was actually invoked (for reporting)
    """

    question: str
    original_question: str
    generation: str
    documents: List[str]
    web_search_needed: str
    loop_count: int
    used_web_search: bool
