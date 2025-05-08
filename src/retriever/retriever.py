import pickle

import hnswlib
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = hnswlib.Index(space='cosine', dim=384)
index.load_index("embeddings/index.bin")
with open("embeddings/knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

def get_similar_responses(question: str, num_responses: int) -> list:

    # 1 Convert Question to Embedding
    embedding = model.encode(question)

    # 2 Compute Similarity of Question to knowledge base (ideally approx nearest neighbor, HNSW)
    # 3 Prune Top K
    labels, distances = index.knn_query(embedding, k=num_responses, filter=None)

    # 4 Get Raw Text
    # 5 Format Output
    results = [knowledge_base[i]["prompt"] for i in labels[0]]

    return results
