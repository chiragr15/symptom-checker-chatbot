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

    def get_disease_predictions(self, user_symptoms, top_k=5):
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
    
    def extract_words_and_phrases_from_sentence(self, sentence, threshold=80):

        NEGATION_WORDS = {"not","no","never","nothing",
            "don't","dont","didn't","didnt",
            "isn't","isnt","wasn't","wasnt",
            "aren't","arent","can't","cant",
            "couldn't","couldnt","won't","wont",
            "shouldn't","shouldnt","wouldn't","wouldnt",
            "haven't","havent","hasn't","hasnt","hadn't","hadnt"}

        TOKEN_RE   = re.compile(r"\b\w+(?:'\w+)?\b")

        vocab        = self.symptom_vocab_list
        single_words = [v for v in vocab if "_" not in v]
        phrases      = [v for v in vocab if "_" in v]

        matched_words, matched_phrases = [], []

        # split text into clauses by punctuation that usually ends / separates thoughts
        clauses = re.split(r"[.;,:!?]", sentence.lower())

        for clause in clauses:
            if not clause.strip():          # skip empty splits
                continue

            tokens = TOKEN_RE.findall(clause)

            # Check if clause is negated
            clause_negated = any(tok in NEGATION_WORDS for tok in tokens)
            if clause_negated:
                continue                    # skip everything in a negated clause

            # single‑word matching
            for tok in tokens:
                hit = process.extractOne(tok, single_words, scorer=fuzz.ratio)
                if hit and hit[1] >= threshold:
                    matched_words.append(hit[0])

            # multi‑word phrase matching (order‑free)
            for phrase in phrases:
                parts = phrase.split('_')
                # each part must fuzzy‑match some token in this clause
                ok = True
                for part in parts:
                    best = process.extractOne(part, tokens, scorer=fuzz.ratio)
                    if not best or best[1] < threshold:
                        ok = False
                        break
                if ok:
                    matched_phrases.append(phrase)
                    # remove constituent single‑word matches we already counted
                    for part in parts:
                        if part in matched_words:
                            matched_words.remove(part)

        # deduplicate (in case the same symptom appears in several clauses)
        matched_words   = list(dict.fromkeys(matched_words))
        matched_phrases = list(dict.fromkeys(matched_phrases))

        return matched_words + matched_phrases

########################### UNIT TEST #####################################

if __name__ == "__main__":
    model = SymptomRetrievalModel(cache_embeddings=False)  # Force regeneration of embeddings
    print(" Symptom-to-Disease Retrieval Tool (type 'exit' to quit)")

    while True:
        user_input = input("\nEnter symptoms:\n")
        if user_input.lower() in ['exit', 'quit']:
            break

        user_symptoms = model.extract_words_and_phrases_from_sentence(user_input)

        print("Matched symptoms:\n")
        print(user_symptoms)

        predictions = model.get_disease_predictions(user_symptoms)
        if not predictions:
            print(" No matching symptoms or diseases found. Try rephrasing.")
        else:
            print("\nTop predicted diseases:")
            for p in predictions:
                print(f" {p['disease']} — matched with '{p['matched_symptom']}' (confidence: {p['confidence']}% - {p['confidence_level']})")

