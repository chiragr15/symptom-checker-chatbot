# symptom_retrieval.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
from rapidfuzz import process, fuzz
import re


class SymptomRetrievalModel:
    def __init__(self, data_path="data/cleaned_symptom_disease.csv", cache_embeddings=True):
        self.df = pd.read_csv(data_path).drop_duplicates(subset=['Symptom', 'Disease'])
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_path = "data/symptom_embeddings.pkl"
        self.cache_embeddings = cache_embeddings

        # Create list of unique symptoms
        self.unique_symptoms = self.df['Symptom'].unique().tolist()

        # Load or compute embeddings
        if self.cache_embeddings and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.symptom_embeddings = pickle.load(f)
        else:
            self.symptom_embeddings = self.model.encode(self.unique_symptoms, convert_to_tensor=True)
            if self.cache_embeddings:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.symptom_embeddings, f)

        # Mapping from symptom to associated diseases
        self.symptom_to_disease = self.df.groupby("Symptom")["Disease"].apply(list).to_dict()

    def correct_symptom_spelling(self, symptom, threshold=80):
        """Correct user input using fuzzy matching"""
        match, score, _ = process.extractOne(symptom, self.unique_symptoms, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return match
        return None  # No confident match

    def preprocess_input(self, raw_input):
        """Normalize multi-line and comma-separated input into a list of symptoms"""
        raw_input = raw_input.replace('\n', ',').replace('\r', ',')
        tokens = [s.strip().lower() for s in raw_input.split(',') if s.strip()]
        corrected = [self.correct_symptom_spelling(s) for s in tokens]
        return [s for s in corrected if s is not None]

    def get_disease_predictions(self, user_input, top_k=5):
        user_symptoms = self.preprocess_input(user_input)

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
        user_input = input("\nEnter symptoms (comma or newline-separated):\n")
        if user_input.lower() in ['exit', 'quit']:
            break

        predictions = model.get_disease_predictions(user_input)
        if not predictions:
            print(" No matching symptoms or diseases found. Try rephrasing.")
        else:
            print("\nTop predicted diseases:")
            for p in predictions:
                print(f" {p['disease']} — matched with '{p['matched_symptom']}' (confidence: {p['confidence']}% - {p['confidence_level']})")

