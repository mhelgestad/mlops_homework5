from pydantic import BaseModel
from typing import List


class RAGRequest(BaseModel):
    question: str
    num_responses: int


class RAGResponseItem(BaseModel):
    question: str
    wiki_excerpt: str


class RAGResponse(BaseModel):
    answers: List[RAGResponseItem]
