# followup.py
import json

with open("data/followup_questions.json") as f:
    followup_questions = json.load(f)

def get_followup_questions(symptom, max_qs=3):
    return followup_questions.get(symptom.lower(), [])[:max_qs]
