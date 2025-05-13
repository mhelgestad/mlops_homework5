import hnswlib
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os

# create embeddings dir if doesn't exist
EMBEDDING_DIR = "embeddings/"
if not os.path.exists(EMBEDDING_DIR):
    os.mkdir(EMBEDDING_DIR)

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load CSV
data = pd.read_csv("data/6000_all_categories_questions_with_excerpts.csv")
questions = data["prompt"].tolist()

# Generate embeddings
embeddings = model.encode(questions, show_progress_bar=True)

index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
index.init_index(max_elements=len(questions), ef_construction=200, M=16)
index.add_items(embeddings)
index.save_index("embeddings/index.bin")

# Save metadata
with open("embeddings/knowledge_base.pkl", "wb") as f:
    pickle.dump(data.to_dict(orient="records"), f)
