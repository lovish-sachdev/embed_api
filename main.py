from fastapi import FastAPI
from pydantic import BaseModel
from onnx_infer import get_embedding, cosine_similarity

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

class ScoreRequest(BaseModel):
    text1: str
    text2: str

@app.post("/embed")
def embed_text(req: EmbedRequest):
    embedding = get_embedding(req.text)
    return {"embedding": list(embedding)}  # Convert NumPy array to Python list

@app.post("/score")
def compare_texts(req: ScoreRequest):
    emb1 = get_embedding(req.text1)
    emb2 = get_embedding(req.text2)
    score = cosine_similarity(emb1, emb2)
    return {"similarity_score": float(score)}  # Ensure it's a native float


# from fastapi import FastAPI
# from pydantic import BaseModel
from vectordb import *

# app = FastAPI()

class AddRequest(BaseModel):
    id: str
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/add")
def add_embedding(req: AddRequest):
    upload_text_embedding(req.id, req.text)
    return {"message": f"Uploaded embedding for ID: {req.id}"}

@app.post("/search")
def search_embedding(req: SearchRequest):
    ids = search_similar_ids(req.query, req.top_k)
    return {"similar_ids": ids}
