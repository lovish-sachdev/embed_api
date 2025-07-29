import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from typing import List

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
session = ort.InferenceSession("model.onnx")

def get_embedding(text: str) -> List[float]:
    """
    Returns a list of floats as embedding for given input text.
    """
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run ONNX model
    last_hidden_state = session.run(None, onnx_inputs)[0]  # shape: (1, seq_len, hidden_dim)

    # Mean pooling → shape: (hidden_dim,) and convert to Python list
    return last_hidden_state.mean(axis=1)[0].tolist()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes cosine similarity between two vectors represented as Python lists.
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))
