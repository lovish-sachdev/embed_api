<<<<<<< HEAD
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
session = ort.InferenceSession("model.onnx")

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    last_hidden_state = session.run(None, onnx_inputs)[0]  # shape: (1, seq_len, hidden_dim)
    return last_hidden_state.mean(axis=1)[0]  # shape: (hidden_dim,)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))
=======
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
session = ort.InferenceSession("model.onnx")

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    last_hidden_state = session.run(None, onnx_inputs)[0]  # shape: (1, seq_len, hidden_dim)
    return last_hidden_state.mean(axis=1)[0]  # shape: (hidden_dim,)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))
>>>>>>> 7e9d4cd37e37a58479d750d0c96655e1218dbb4f
