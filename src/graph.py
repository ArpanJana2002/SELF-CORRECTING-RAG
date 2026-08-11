"""
Wires the nodes from src/nodes.py into a LangGraph StateGraph implementing
Corrective-RAG style self-correction:

                         ┌─────────────┐
                         │  retrieve   │
                         └──────┬──────┘
                                ▼
                       ┌─────────────────┐
                       │ grade_documents │
                       └────────┬────────┘
                                ▼
                    (decide_to_generate router)
                 ┌──────────────┴───────────────┐
                 │ enough relevant docs?         │ not enough?
                 ▼                               ▼
           ┌──────────┐                 ┌─────────────────┐
           │ generate │◄────────────────┤  transform_query│
           └────┬─────┘                 └────────┬────────┘
                │                                 ▼
                │                          ┌─────────────┐
                │                          │ web_search  │
                │                          └──────┬──────┘
                │                                 │
                │◄────────────────────────────────┘
                ▼
     (grade_generation router)
   ┌───────────┬─────────────────┬───────────────┐
   │ useful    │ not supported   │ not useful     │
   ▼           ▼ (hallucinated)  ▼ (off-topic)
  END     back to generate   back to transform_query
"""
from langgraph.graph import StateGraph, END

from src.state import GraphState
from src import nodes


def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("generate", nodes.generate)

    # Entry point
    workflow.set_entry_point("retrieve")

    # Edges
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation,
        {
            "useful": END,
            "not useful": "transform_query",
            "not supported": "generate",
        },
    )

    return workflow.compile()


def run(question: str, verbose: bool = True):
    """Convenience entry point: run the graph end-to-end for a single question."""
    app = build_graph()
    initial_state = {
        "question": question,
        "original_question": question,
        "generation": "",
        "documents": [],
        "web_search_needed": "No",
        "loop_count": 0,
        "used_web_search": False,
    }

    state = dict(initial_state)
    for step_output in app.stream(initial_state):
        for node_name, node_update in step_output.items():
            if verbose:
                print(f"\n[node completed: {node_name}]")
            state.update(node_update)

    return state
