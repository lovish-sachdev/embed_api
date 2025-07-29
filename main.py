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
    return {"embedding": embedding}

@app.post("/score")
def compare_texts(req: ScoreRequest):
    emb1 = get_embedding(req.text1)
    emb2 = get_embedding(req.text2)
    score = cosine_similarity(emb1, emb2)
    return {"similarity_score": score}
