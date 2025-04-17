
##!pip install rapidfuzz

import pandas as pd
from rapidfuzz import process, fuzz

class SymptomSeverityChecker:
    def __init__(self, severity_data_path="data/cleaned_symptom_severity.csv"):
        self.df = pd.read_csv(severity_data_path)
        self.symptoms = self.df['Symptom'].str.lower().tolist()
        self.severity_map = dict(zip(self.df['Symptom'].str.lower(), self.df['SeverityLevel'].str.lower()))

    def correct_symptom_spelling(self, symptom, threshold=80):
        """Fuzzy match symptom input to known list after normalizing format"""
        normalized = symptom.lower().strip().replace(" ", "_")
        match, score, _ = process.extractOne(normalized, self.symptoms, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return match
        return None

    def preprocess_input(self, raw_input):
        """Normalize multi-line or comma-separated input into clean symptom list"""
        raw_input = raw_input.replace('\n', ',').replace('\r', ',')
        tokens = [s.strip().lower() for s in raw_input.split(',') if s.strip()]
        corrected = [self.correct_symptom_spelling(s) for s in tokens]
        return [s for s in corrected if s is not None]

    def classify_severity(self, user_input):
        symptoms = self.preprocess_input(user_input)

        if not symptoms:
            return []

        results = []
        for symptom in symptoms:
            severity = self.severity_map.get(symptom, "unknown")
            results.append({
                "symptom": symptom,
                "severity": severity,
                "alert": "Seek immediate medical attention." if severity == "severe"
                         else "ℹTake precautions and monitor."
                         if severity in ["moderate", "mild"] else "Unknown severity."
            })
        return results


################################################################

if __name__ == "__main__":
    checker = SymptomSeverityChecker()
    print(" Symptom Severity Classifier (type 'exit' to quit)")

    while True:
        user_input = input("\nDescribe your symptom(s):\n")
        if user_input.lower() in ['exit', 'quit']:
            break

        results = checker.classify_severity(user_input)
        if not results:
            print(" No known symptoms matched. Please try again.")
        else:
            print("\nSeverity Analysis:")
            for r in results:
                print(f" {r['symptom'].replace('_',' ').title()} — Severity: {r['severity'].capitalize()} → {r['alert']}")
