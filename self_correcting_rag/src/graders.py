"""
The "self-correction" brains of the agent: a set of small LLM calls that each
return a structured verdict, used by the graph to decide what to do next.

  - GradeDocuments   -> is this retrieved chunk actually relevant to the question?
  - GradeHallucinations -> is the generated answer grounded in the retrieved facts?
  - GradeAnswer      -> does the generated answer actually resolve the question?
  - rewrite_question -> produce a better-phrased query when retrieval/generation fails
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src import config


# ---------- Structured schemas ----------

class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: str = Field(
        description="Is the document relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for whether the generation is grounded in the given facts."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for whether the generation actually answers the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# ---------- Graders (each is a small prompt + structured-output LLM call) ----------

def get_document_grader():
    llm = config.get_llm(temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )
    return prompt | structured_llm


def get_hallucination_grader():
    llm = config.get_llm(temperature=0)
    structured_llm = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means the answer is grounded in / supported by the facts."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
        ]
    )
    return prompt | structured_llm


def get_answer_grader():
    llm = config.get_llm(temperature=0)
    structured_llm = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question:\n\n{question}\n\nLLM generation: {generation}"),
        ]
    )
    return prompt | structured_llm


def get_question_rewriter():
    llm = config.get_llm(temperature=0)

    system = """You are a question re-writer that converts an input question to a better version that is
optimized for vectorstore retrieval and/or web search. Look at the input and try to reason about the
underlying semantic intent / meaning. Return ONLY the rewritten question, nothing else."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question:\n\n{question}\n\nFormulate an improved question.",
            ),
        ]
    )
    return prompt | llm
