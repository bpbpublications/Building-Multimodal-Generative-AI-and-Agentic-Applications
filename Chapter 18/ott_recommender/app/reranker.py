# ott_recommender/app/reranker.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_candidates=10):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.max_candidates = max_candidates

    def rank(self, query, candidates):
        # Limit to safe number of candidates
        candidates = candidates[:self.max_candidates]
        pairs = [(query, cand) for cand in candidates]

        scores = []
        for query_text, cand_text in pairs:
            inputs = self.tokenizer(query_text, cand_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                score = self.model(**inputs).logits.squeeze().item()
                scores.append(score)

        return [c for _, c in sorted(zip(scores, candidates), reverse=True)]

