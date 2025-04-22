from sympton_retrieval import SymptomRetrievalModel
from symptom_severity_checker import SymptomSeverityChecker
from followup import get_followup_questions
from symptom_utils import extract_symptoms_from_sentence

def main():
    print("🩺 Symptom Checker Chatbot (CLI Mode)")
    print("Type 'exit' to quit\n")

    retriever = SymptomRetrievalModel()
    severity_checker = SymptomSeverityChecker()

    while True:
        user_input = input(" Describe your symptoms: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        # Extract initial symptoms
        session_symptoms = extract_symptoms_from_sentence(user_input, retriever.symptom_vocab_list)
        if not session_symptoms:
            print("⚠️ No recognizable symptoms found. Try again.\n")
            continue

        print(f"\n Detected symptoms: {session_symptoms}")

        # Step 1: Predict initial diseases
        disease_results = retriever.get_disease_predictions(user_input)
        if not disease_results:
            print("⚠️ No disease predictions found.\n")
            continue

        print("\n Top Predicted Conditions:")
        for res in disease_results:
            print(f" {res['disease']} — matched with '{res['matched_symptom']}' (confidence: {res['confidence']}% - {res['confidence_level']})")

        # Step 2: Ask follow-up based on top matched symptom
        top_symptom = disease_results[0]['matched_symptom']
        followups = get_followup_questions(top_symptom)

        if followups:
            print(f"\n Follow-up questions for '{top_symptom}':")
            for q in followups:
                print(f" {q}")

            followup_answer = input("\n🧍 Your answer: ").strip()
            if followup_answer:
                new_symptoms = extract_symptoms_from_sentence(followup_answer, retriever.symptom_vocab_list)

                if new_symptoms:
                    print(f" Additional symptoms detected: {new_symptoms}")
                    session_symptoms.extend(new_symptoms)
                    session_symptoms = list(set(session_symptoms))  # remove duplicates
                else:
                    print(" No new symptoms detected in your response.")

        # Step 3: Refine prediction based on updated symptom list
        refined_input = ", ".join(session_symptoms)
        refined_results = retriever.get_disease_predictions(refined_input)

        print("\n Refined Disease Predictions:")
        for res in refined_results:
            print(f" {res['disease']} — matched with '{res['matched_symptom']}' (confidence: {res['confidence']}% - {res['confidence_level']})")

        # Step 4: Run severity analysis
        severity_results = severity_checker.classify_severity(refined_input)
        print("\n Severity Assessment:")
        for res in severity_results:
            print(f" {res['symptom']} — Severity: {res['severity'].capitalize()} → {res['alert']}")

        print("\n You can enter new symptoms to restart, or type 'exit' to quit.\n")

if __name__ == "__main__":
    main()
