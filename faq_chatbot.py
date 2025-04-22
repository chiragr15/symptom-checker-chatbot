import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FAQChatbot:
    def __init__(self, faq_csv_path="data/faq_dataset.csv"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = pd.read_csv(faq_csv_path).dropna()
        self.questions = self.df['Question'].tolist()
        self.answers = self.df['Answer'].tolist()
        
        # Precompute question embeddings
        self.question_embeddings = self.model.encode(
            self.questions, convert_to_tensor=False
        )

    def get_best_match(self, user_query, top_k=1):
        user_embedding = self.model.encode([user_query], convert_to_tensor=False)
        sims = cosine_similarity(user_embedding, self.question_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": sims[idx]
            })
        return results
