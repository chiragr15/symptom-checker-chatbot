import pandas as pd
import re
import json
from collections import defaultdict
from pathlib import Path

# Step 1: Load full symptom list
def load_symptom_list():
    vocab_file = Path("data/symptom_vocab.txt")
    if vocab_file.exists():
        with open(vocab_file) as f:
            return sorted(set(line.strip().lower() for line in f if line.strip()))
    else:
        df = pd.read_csv("data/cleaned_symptom_disease.csv")
        return sorted(set(df['Symptom'].str.lower().dropna().tolist()))

# Step 2: Load Q&A dataset
def load_question_data():
    return pd.read_parquet("data/medical_q_a.parquet")

# Step 3: Patterns to identify follow-up-like questions
followup_patterns = [
    r"\bhow long\b",
    r"\bdo you (also|still|have|experience|feel)?\b",
    r"\bare you\b",
    r"\bis it\b",
    r"\bdoes it\b",
    r"\bhave you been\b",
    r"\bare there\b"
]

# Step 4: Curated fallback questions for common symptoms
curated_fallbacks = {
    "fever": [
        "Have you measured your temperature?",
        "Do you have chills or sweating?",
        "Is your fever continuous or intermittent?"
    ],
    "headache": [
        "Is your headache on one side or both?",
        "Do you feel nauseous or dizzy with the headache?",
        "How long has it lasted?"
    ],
    "nausea": [
        "Have you vomited or just felt sick?",
        "Is it worse after eating?",
        "Is the nausea constant or on and off?"
    ],
    "chest pain": [
        "Is the pain sharp, dull, or burning?",
        "Does it get worse with breathing or movement?",
        "Are you feeling short of breath?"
    ],
    "rash": [
        "Is the rash itchy or painful?",
        "Did it spread quickly?",
        "Do you also have a fever?"
    ]
}

# Step 5: Template fallback generator
def generate_template_followups(symptom):
    return [
        f"How long have you had {symptom}?",
        f"Is your {symptom} constant or does it come and go?",
        f"Have you experienced anything else along with the {symptom}?"
    ]

# Step 6: Clean a question
def clean_question(text):
    text = re.sub(r'^(hi|hello|dear) doctor[:,]?\s*', '', text.strip(), flags=re.IGNORECASE)
    if '?' in text:
        text = text.split('?')[0] + '?'
    return text.strip()

# Step 7: Build hybrid follow-up dictionary
def build_followup_dict():
    all_symptoms = load_symptom_list()
    df = load_question_data()
    symptom_qs = defaultdict(list)

    for q in df['input'].dropna():
        q = q.strip()
        q_lower = q.lower()

        if len(q) > 250 or q.count('.') > 2:
            continue
        if not any(re.search(p, q_lower) for p in followup_patterns):
            continue

        for sym in all_symptoms:
            if sym in q_lower:
                cleaned = clean_question(q)
                if cleaned.endswith('?') and len(cleaned.split()) >= 5:
                    symptom_qs[sym].append(cleaned)

    # Final dictionary
    final_dict = {}

    for sym in all_symptoms:
        if sym in symptom_qs and symptom_qs[sym]:
            final_dict[sym] = symptom_qs[sym][:3]  # Real extracted
        elif sym in curated_fallbacks:
            final_dict[sym] = curated_fallbacks[sym]  # Curated
        else:
            final_dict[sym] = generate_template_followups(sym)  # Generated

    with open("data/followup_questions.json", "w") as f:
        json.dump(final_dict, f, indent=2)

    print(f"✅ Hybrid followup_questions.json generated with {len(final_dict)} symptoms covered.")

# Run
if __name__ == "__main__":
    build_followup_dict()
