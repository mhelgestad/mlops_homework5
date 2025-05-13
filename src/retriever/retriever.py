import pickle

import hnswlib
import numpy
from sentence_transformers import SentenceTransformer

from src.models.query import RAGResponseItem

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = hnswlib.Index(space="cosine", dim=384)
index.load_index("embeddings/index.bin")
with open("embeddings/knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)


def get_similar_responses(question: str, num_responses: int) -> list:

    # 1 Convert Question to Embedding
    embedding = _convert_question(question)

    # 2 Compute Similarity of Question to knowledge base (ideally approx nearest neighbor, HNSW)
    # 3 Prune Top K
    labels = _compute_similarity(embedding, num_responses)

    # 4 Get Raw Text
    # 5 Format Output
    return _format_results(labels)


def _convert_question(question: str) -> numpy.ndarray:
    return model.encode(question)


def _compute_similarity(embedding: numpy.ndarray, num_responses: int) -> tuple:
    labels, _ = index.knn_query(embedding, k=num_responses, filter=None)
    return labels


def _format_results(labels) -> list[RAGResponseItem]:
    results = []
    for i in labels[0]:
        kbi = knowledge_base[i]
        item = RAGResponseItem(
            question=kbi["prompt"], wiki_excerpt=kbi["wikipedia_excerpt"]
        )
        results.append(item)
    return results
