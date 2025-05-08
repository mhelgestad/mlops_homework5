from pydantic import BaseModel
from typing import List

class RAGRequest(BaseModel):
    question: str
    num_responses: int

class RAGResponse(BaseModel):
    answers: List[str]
