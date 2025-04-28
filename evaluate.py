import json
import time
from sympton_retrieval import SymptomRetrievalModel
from symptom_severity_checker import SymptomSeverityChecker
from followup import followup_questions


retriever = SymptomRetrievalModel(cache_embeddings=True)
severity_checker = SymptomSeverityChecker()

# Test Symptom-pairs
test_cases = [
    ("headache", ["migraine", "hypertension"]),
    ("vomiting", ["gastroenteritis", "peptic ulcer disease"]),
    ("rash", ["chickenpox", "measles"]),
    ("nausea", ["hepatitis a", "hepatitis b"]),
    ("chills", ["malaria", "dengue"]),
    ("fever", ["malaria", "flu", "typhoid"]),
    ("cough", ["bronchitis", "pneumonia"]),
    ("diarrhoea", ["gastroenteritis", "cholera"]),
]

#Metrics Counters
retrieval_success = 0
retrieval_total = len(test_cases)
retrieval_times = []

#Top-3 Accuracy
for symptom_text, expected_diseases in test_cases:
    start_time = time.time()

    results = retriever.get_disease_predictions(symptom_text)
    top3 = [res['disease'].lower() for res in results[:3]]

    retrieval_times.append(time.time() - start_time)

    if any(disease.lower() in top3 for disease in expected_diseases):
        retrieval_success += 1

retrieval_top3_accuracy = (retrieval_success / retrieval_total) * 100
average_retrieval_time = sum(retrieval_times) / retrieval_total

# Static testing, no real metric
test_symptoms_for_severity = ["headache", "vomiting", "rash", "nausea", "chills", "fever", "cough", "diarrhoea"]
severity_mapping_success = 0

for symptom in test_symptoms_for_severity:
    severity = severity_checker.classify_severity(symptom)
    if severity:
        severity_mapping_success += 1

severity_mapping_rate = (severity_mapping_success / len(test_symptoms_for_severity)) * 100

#Follow-Up Question Coverage
symptom_vocab_size = len(retriever.symptom_vocab_list)
followup_coverage = (len(followup_questions) / symptom_vocab_size) * 100

print("\n---------- WellWise Evaluation Summary ------------")
print(f"Symptom-to-Disease Retrieval Top-3 Accuracy: {retrieval_top3_accuracy:.2f}%")
print(f"Average Retrieval Time per Query: {average_retrieval_time*1000:.2f} ms")
print(f"Severity Mapping Success (static lookup): {severity_mapping_rate:.2f}%")
print(f"Follow-Up Question Symptom Coverage: {followup_coverage:.2f}%")
print("==========================================\n")
