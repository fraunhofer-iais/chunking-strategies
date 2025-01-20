from typing import Optional, List, Dict

from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int


class Answer(BaseModel):
    answer: str
    start: Optional[int] = None
    end: Optional[int] = None
    spans: Optional[list[Span]] = None


class EvalSample(BaseModel):
    document_id: str
    document: str
    questions: Optional[List[str]] = None
    answers: Optional[List[Answer]] = None


class RetrievedParagraphs(BaseModel):
    document_id: str
    question: str
    paragraphs: List[str]
    scores: List[float]


class RetrieverResult(BaseModel):
    map: float
    mrr: float
    detailed_summary: List[Dict]
    relevance_indicators: List


class RetrieverResults(BaseModel):
    map_documents: float
    mrr_documents: float
    per_eval_sample_results: List[RetrieverResult]