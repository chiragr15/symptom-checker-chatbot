# symptom_utils.py

import re
from rapidfuzz import process, fuzz

NEGATION_WORDS = {"not","no","never","nothing",
    "don't","dont","didn't","didnt",
    "isn't","isnt","wasn't","wasnt",
    "aren't","arent","can't","cant",
    "couldn't","couldnt","won't","wont",
    "shouldn't","shouldnt","wouldn't","wouldnt",
    "haven't","havent","hasn't","hasnt","hadn't","hadnt"}

TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b")

def split_into_clauses(text: str):
    return re.split(r"[.;,:!?]", text.lower())

def tokenize(text: str):
    return TOKEN_RE.findall(text)

def match_single_words(tokens: list, single_words: list, threshold=80):
    matches = []
    for tok in tokens:
        hit = process.extractOne(tok, single_words, scorer=fuzz.ratio)
        if hit and hit[1] >= threshold:
            matches.append(hit[0])
    return matches

def match_phrases(tokens: list, phrases: list, threshold=80):
    matches = []
    for phrase in phrases:
        parts = phrase.split('_')
        ok = True
        for part in parts:
            best = process.extractOne(part, tokens, scorer=fuzz.ratio)
            if not best or best[1] < threshold:
                ok = False
                break
        if ok:
            matches.append(phrase)
    return matches

def extract_symptoms_from_sentence(sentence: str, vocab: list, threshold=80):
    single_words = [v for v in vocab if "_" not in v]
    phrases = [v for v in vocab if "_" in v]

    matched_words, matched_phrases = [], []

    # Synonym normalization (extend this as needed)
    synonym_map = {
        "vomited": "vomiting",
        "dizzy": "dizziness",
        "dizziness": "dizziness",
        "nauseous": "nausea",
        "headaches": "headache",
        "coughing": "cough",
        "fevers": "fever",
        "palpitations": "palpitation"
    }

    # Normalize synonyms
    sentence_lower = sentence.lower()
    for alt, std in synonym_map.items():
        sentence_lower = sentence_lower.replace(alt, std)

    for clause in split_into_clauses(sentence_lower):
        if not clause.strip():
            continue

        tokens = tokenize(clause)

        if any(tok in NEGATION_WORDS for tok in tokens):
            continue

        word_matches = match_single_words(tokens, single_words, threshold)
        matched_words.extend(word_matches)
        phrase_matches = match_phrases(tokens, phrases, threshold)
        matched_phrases.extend(phrase_matches)

        for phrase in phrase_matches:
            for part in phrase.split('_'):
                if part in matched_words:
                    matched_words.remove(part)

    matched_words = list(dict.fromkeys(matched_words))
    matched_phrases = list(dict.fromkeys(matched_phrases))

    return matched_words + matched_phrases
