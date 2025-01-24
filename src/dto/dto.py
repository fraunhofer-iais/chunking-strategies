from typing import Optional, List, Dict

from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int


class Answer(BaseModel):
    answer: str
    start: Optional[int] = None
    end: Optional[int] = None


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
    detailed_summary: List[Dict]
    map: float  # mean_average_precision
    mrr: float  # mrr over all questions in a document
    relevance_indicators: List


class RetrieverResults(BaseModel):
    map_documents: float  # average of map over documents
    mrr_documents: float  # average of mrr over documents
    total_documents: int
    total_questions: int
    average_questions_per_document: float
    per_eval_sample_results: List[RetrieverResult]
