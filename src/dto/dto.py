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


class RetrieverResult(BaseModel):
    document_id: str
    question: str
    paragraphs: List[str]
    scores: List[float]


class EvalResult(BaseModel):
    recall_at_k: List[float]


class DocumentEvalResult(EvalResult):
    eval_sample: EvalSample
    retriever_results: List[RetrieverResult]


class QuestionEvalResult(EvalResult):
    document_id: str
    question: str
    

class AverageDocResult(BaseModel):
    average_recall_at_k: List[float]
