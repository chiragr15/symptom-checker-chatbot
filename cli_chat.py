import sys
from sympton_retrieval import SymptomRetrievalModel
from faq_chatbot import FAQChatbot
from symptom_utils import extract_symptoms_from_sentence

def main():
    symptom_model = SymptomRetrievalModel(
        data_path="data/cleaned_symptom_disease.csv",
        symptom_vocab_path="data/symptom_vocabulary.csv",
        cache_embeddings=False
    )
    faq_model = FAQChatbot("data/faq_dataset.csv")

    recognized_symptoms = set()
    diagnosis_complete = False
    
    print("Welcome to the CLI Medical Symptom Checker!")
    print("Type 'exit' to quit at any time.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            sys.exit(0)
        
        if diagnosis_complete:
            # After final diagnosis, any new input is treated as FAQ
            handle_faq_query(faq_model, user_input)
            continue

        # If user says no more symptoms
        stop_words = ["no", "nope", "none", "nothing else", "that's all", "done"]
        if any(s in user_input.lower() for s in stop_words):
            finalize_diagnosis(symptom_model, recognized_symptoms)
            diagnosis_complete = True
            continue

        # Attempt to extract recognized symptoms
        new_symptoms = extract_symptoms_from_sentence(
            user_input, symptom_model.symptom_vocab_list
        )
        if new_symptoms:
            for s in new_symptoms:
                recognized_symptoms.add(s)
            # Check confidence
            diagnosis_complete = check_confidence_and_ask_followups(
                symptom_model, recognized_symptoms
            )
        else:
            # No recognized symptoms => FAQ
            handle_faq_query(faq_model, user_input)

def handle_faq_query(faq_model, user_input):
    faq_res = faq_model.get_best_match(user_input, top_k=1)
    if faq_res and faq_res[0]["score"] > 0.5:
        print(f"\nChatbot (FAQ): {faq_res[0]['answer']}")
    else:
        print("\nChatbot (FAQ): I'm not sure about that. Please try rephrasing or list symptoms.")

def check_confidence_and_ask_followups(symptom_model, recognized_symptoms):
    if not recognized_symptoms:
        ask_for_more_symptoms()
        return False
    
    text = " ".join(recognized_symptoms)
    predictions = symptom_model.get_disease_predictions(text, top_k=3)
    if not predictions:
        ask_for_more_symptoms()
        return False
    
    top_pred = predictions[0]
    if top_pred["confidence_level"] in ["High", "Very High"]:
        finalize_diagnosis(symptom_model, recognized_symptoms)
        return True
    else:
        ask_for_more_symptoms()
        return False

def ask_for_more_symptoms():
    print(
        "\nChatbot: Could you describe any other symptoms you have?"
        " Or say 'no' if that's all."
    )

def finalize_diagnosis(symptom_model, recognized_symptoms):
    if not recognized_symptoms:
        print("\nChatbot: No recognized symptoms, so I can't make a suggestion.")
        return
    
    text = " ".join(recognized_symptoms)
    preds = symptom_model.get_disease_predictions(text)
    if not preds:
        print("\nChatbot: I'm not sure I understand your symptoms. Try again later.")
        return

    print("\nChatbot: Based on your symptoms, possible conditions include:")
    for p in preds[:3]:
        print(
            f" - {p['disease']} (matched '{p['matched_symptom']}'), "
            f"confidence: {p['confidence']}% ({p['confidence_level']})"
        )
    print(
        "\n⚠️ Disclaimer: This is an AI-powered tool for informational purposes only, "
        "not a substitute for professional medical advice."
    )

if __name__ == "__main__":
    main()
