##Install requirements
##!pip install rapidfuzz

import pandas as pd
from rapidfuzz import process, fuzz
import re
from symptom_utils import extract_symptoms_from_sentence

class SymptomSeverityChecker:
    def __init__(self, severity_data_path="data/cleaned_symptom_severity.csv"):
        self.df = pd.read_csv(severity_data_path)
        self.symptoms = self.df['Symptom'].str.lower().tolist()
        self.severity_map = dict(zip(self.df['Symptom'].str.lower(), self.df['SeverityLevel'].str.lower()))
        self.symptom_vocab_list = self.symptoms

    def classify_severity(self, symptoms):
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
    
    def extract_symptoms_from_sentence(self, sentence, threshold=80):
        return extract_symptoms_from_sentence(sentence, self.symptom_vocab_list, threshold)


################################################################

if __name__ == "__main__":
    checker = SymptomSeverityChecker()
    print(" Symptom Severity Classifier (type 'exit' to quit)")

    while True:
        user_input = input("\nDescribe your symptom(s):\n")
        if user_input.lower() in ['exit', 'quit']:
            break

        user_symptoms = checker.extract_symptoms_from_sentence(user_input)

        print("Matched symptoms:\n")
        print(user_symptoms)

        results = checker.classify_severity(user_symptoms)
        if not results:
            print(" No known symptoms matched. Please try again.")
        else:
            print("\nSeverity Analysis:")
            for r in results:
                print(f" {r['symptom'].replace('_',' ').title()} — Severity: {r['severity'].capitalize()} → {r['alert']}")
