import os
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

FAISS_DIR = "api/faiss"
METAS_FILE = "api/faiss/index.json"

# 1️⃣ Load your metas.json file
with open(METAS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["chunk"] for item in data]
metas = [item.get("meta", {}) for item in data]

print(f"✅ Loaded {len(texts)} chunks")

# 2️⃣ Use FastEmbed (no torch, no GPU)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")

print("⏳ Creating FAISS index...")
db = FAISS.from_texts(
    texts,
    embedding=embeddings,
    metadatas=metas
)

# 3️⃣ Ensure faiss directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

# 4️⃣ Save index
db.save_local(FAISS_DIR)

print("✅ FAISS index created successfully!")
print(f"✅ Files saved inside: {FAISS_DIR}")
print("✅ index.faiss, index.pkl, vectors.npy, metas.json generated!")
