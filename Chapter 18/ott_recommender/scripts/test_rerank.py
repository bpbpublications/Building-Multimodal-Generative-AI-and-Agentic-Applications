from app.reranker import Reranker

def test():
    query = "I like action and thriller movies."
    candidates = [
        "Fast & Furious on Netflix",
        "Cooking Show on Disney+",
        "Mindhunter on Amazon Prime"
    ]
    reranker = Reranker()
    ranked = reranker.rank(query, candidates)
    print("Top Recommendation:", ranked[0])

if __name__ == "__main__":
    test()
