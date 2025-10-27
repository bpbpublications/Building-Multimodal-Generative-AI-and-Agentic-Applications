
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def rerank_with_cross_encoder(query, passages, top_k=5):
    pairs = [(query, passage) for passage in passages]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = F.softmax(logits, dim=1)[:, 1]
    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [text for text, score in ranked[:top_k]]
