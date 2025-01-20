from typing import Optional

from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    start: Optional[int] = None
    end: Optional[int] = None


class EvalSample(BaseModel):
    document_id: str
    document: str
    questions: Optional[list[str]] = None
    answers: Optional[list[Answer]] = None


class RetrievedParagraphs(BaseModel):
    document_id: str
    question: str
    paragraphs: list[str]
    scores: list[float]