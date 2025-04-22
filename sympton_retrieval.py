# symptom_retrieval.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
from rapidfuzz import process, fuzz
import re
from symptom_utils import extract_symptoms_from_sentence
from followup import get_followup_questions


class SymptomRetrievalModel:
    def __init__(self, data_path="data/cleaned_symptom_disease.csv", symptom_vocab_path="data/symptom_vocabulary.csv", cache_embeddings=True):
        self.df = pd.read_csv(data_path).drop_duplicates(subset=['Symptom', 'Disease'])
        self.symptom_vocab_list = pd.read_csv(symptom_vocab_path)['Symptom'].tolist()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_path = "data/symptom_embeddings.pkl"
        self.cache_embeddings = cache_embeddings

        # Create list of unique symptoms
        self.unique_symptoms = self.df['Symptom'].unique().tolist()

        # Load or compute embeddings
        if self.cache_embeddings and os.path.exists(self.cache_path):
            self.symptom_embeddings = self.load_pickle(self.cache_path)
        else:
            self.symptom_embeddings = self.model.encode(self.unique_symptoms, convert_to_tensor=True)
            if self.cache_embeddings:
                self.save_pickle(self.symptom_embeddings, self.cache_path)

        # Mapping from symptom to associated diseases
        self.symptom_to_disease = self.df.groupby("Symptom")["Disease"].apply(list).to_dict()

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def get_disease_predictions(self, user_input, top_k=5):
        user_symptoms = extract_symptoms_from_sentence(user_input, self.symptom_vocab_list)
        if not user_symptoms:
            return []  # no valid symptoms after spell correction

        # Embed and average valid user symptoms
        user_embeddings = self.model.encode(user_symptoms, convert_to_tensor=True)
        avg_embedding = np.mean(user_embeddings.cpu().numpy(), axis=0).reshape(1, -1)

        # Compute cosine similarity
        similarities = cosine_similarity(avg_embedding, self.symptom_embeddings.cpu().numpy())[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_symptoms = [self.unique_symptoms[i] for i in top_indices]
        matched_scores = [similarities[i] for i in top_indices]

        results = []
        for symptom, score in zip(matched_symptoms, matched_scores):
            diseases = self.symptom_to_disease.get(symptom, [])
            for disease in diseases:
                raw_score = float(score)
                confidence = round(raw_score * 100)

                if confidence >= 95:
                    level = "Very High"
                elif confidence >= 85:
                    level = "High"
                elif confidence >= 70:
                    level = "Moderate"
                else:
                    level = "Low"

                results.append({
                    "disease": disease.title(),
                    "matched_symptom": symptom,
                    "score": round(raw_score, 3),         # keep for backend
                    "confidence": confidence,             # for display
                    "confidence_level": level             # for chatbot or GUI
                })

        # Deduplicate: Keep only the highest score per disease
        seen = {}
        for item in results:
            if item['disease'] not in seen or item['score'] > seen[item['disease']]['score']:
                seen[item['disease']] = item

        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:top_k]

########################### UNIT TEST #####################################

if __name__ == "__main__":
    model = SymptomRetrievalModel(cache_embeddings=False)  # Force regeneration of embeddings
    print(" Symptom-to-Disease Retrieval Tool (type 'exit' to quit)")

    while True:
        user_input = input("\nEnter symptoms:\n")
        if user_input.lower() in ['exit', 'quit']:
            break

        predictions = model.get_disease_predictions(user_input)
        if not predictions:
            print(" No matching symptoms or diseases found. Try rephrasing.")
        else:
            print("\nTop predicted diseases:")
            for p in predictions:
                print(f" {p['disease']} — matched with '{p['matched_symptom']}' (confidence: {p['confidence']}% - {p['confidence_level']})")
            
            matched_symptom = predictions[0]['matched_symptom']
            followups = get_followup_questions(matched_symptom)

            if followups:
                print("\n🤖 Follow-up questions:")
                for q in followups:
                    print(f"👉 {q}")
