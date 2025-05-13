from fastapi import APIRouter
from src.retriever import retriever
from src.models.query import RAGRequest, RAGResponse

router = APIRouter()


@router.post("/similar_responses", response_model=RAGResponse)
def get_similar_responses(request: RAGRequest):
    results = retriever.get_similar_responses(request.question, request.num_responses)
    return RAGResponse(answers=results)
