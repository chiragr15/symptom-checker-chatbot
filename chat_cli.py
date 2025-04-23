from sympton_retrieval import SymptomRetrievalModel
from symptom_severity_checker import SymptomSeverityChecker
from followup import get_followup_questions
from symptom_utils import extract_symptoms_from_sentence
from faq_chatbot import FAQChatbot


def main():
    retriever = SymptomRetrievalModel(cache_embeddings=False)
    severity_checker = SymptomSeverityChecker()
    faq_model = FAQChatbot("data/faq_dataset.csv")

    print("🩺 Symptom Checker Chatbot (CLI Mode)")
    print("Type 'exit' to quit\n")

    # Store all known symptoms for this entire conversation
    session_symptoms = set()

    # To avoid asking follow-ups repeatedly for the same symptom,
    # we'll track which symptoms we have already asked about.
    asked_followups_for = set()

    while True:
        user_input = input(" Enter symptoms or ask a question: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        # (A) Detect new symptoms from main user input
        new_symptoms = set(extract_symptoms_from_sentence(user_input, retriever.symptom_vocab_list))

        if new_symptoms:
            # Merge into master list
            session_symptoms |= new_symptoms  # set union
            print(f"\n [User Provided New Symptoms] => {new_symptoms}")
            print(f" Current All Symptoms: {session_symptoms}")

            # Run diagnosis & severity, then do follow-ups
            run_diagnosis_and_followups(
                session_symptoms, 
                asked_followups_for, 
                retriever, 
                severity_checker, 
                faq_model
            )
        else:
            # (B) No new symptoms => treat the user input as a FAQ question
            handle_faq_query(faq_model, user_input)

        print("\n(You can continue adding symptoms or ask any question.)\n")


def run_diagnosis_and_followups(
    session_symptoms: set, 
    asked_followups_for: set, 
    retriever, 
    severity_checker, 
    faq_model
):
    """
    1. Runs disease predictions and severity on all known symptoms.
    2. For each symptom that has *not* yet been asked about,
       ask follow-up questions.
    3. If user’s response yields new symptoms, incorporate them
       and loop again (so we keep diagnosing & updating).
    """
    while True:
        # --- 1) DIAGNOSIS ---
        symptom_list_str = ", ".join(session_symptoms)
        disease_results = retriever.get_disease_predictions(symptom_list_str)
        if not disease_results:
            print("\n⚠️ No disease predictions found.")
        else:
            print("\n Current Predicted Conditions (based on all known symptoms):")
            for res in disease_results:
                print(
                    f" - {res['disease']} (matched with '{res['matched_symptom']}') "
                    f"[confidence: {res['confidence']}% - {res['confidence_level']}]"
                )

        # --- 2) SEVERITY ---
        if session_symptoms:
            severity_results = severity_checker.classify_severity(symptom_list_str)
            print("\n Severity Assessment:")
            for sres in severity_results:
                print(
                    f" - {sres['symptom']}: Severity={sres['severity'].capitalize()} → {sres['alert']}"
                )
        else:
            print("\n No known symptoms to assess severity.")

        # --- 3) FOLLOW-UP for each *newly discovered* symptom ---
        # We'll iterate over *all* session symptoms that we've never asked follow-ups about.
        unasked_symptoms = [s for s in session_symptoms if s not in asked_followups_for]

        if not unasked_symptoms:
            # No new symptoms that we haven't asked about => done for now
            break

        for symptom in unasked_symptoms:
            followups = get_followup_questions(symptom)
            # Mark that we've asked about it, so we don’t repeat in the future
            asked_followups_for.add(symptom)

            if not followups:
                continue  # no follow-up Q for this symptom

            print(f"\nFollow-up questions for the newly mentioned symptom '{symptom}':")
            # For simplicity, display them all at once, get a single user response
            for i, q in enumerate(followups, 1):
                print(f"  {i}. {q}")

            user_answer = input("\n🧍 Your answer to the above follow-up questions or please ask any other question you have: ").strip()
            if user_answer.lower() in ["exit", "quit"]:
                # User wants to end chatbot
                exit(0)

            # 3a) Check if user follow-up reveals *new* symptoms
            newly_found = set(extract_symptoms_from_sentence(user_answer, retriever.symptom_vocab_list))
            if newly_found:
                print(f"  [New Follow-up Symptoms] => {newly_found}")
                session_symptoms |= newly_found
                print(f" Current All Symptoms: {session_symptoms}")
            else:
                # If no new symptoms => treat that answer as a FAQ query
                handle_faq_query(faq_model, user_answer)
                break
                # (We do *not* break; we keep going for other unasked symptoms)

        # After we finish *asking* about all unasked symptoms,
        # we re-run the loop if we discovered *additional* new
        # symptoms in these follow-ups. That triggers updated 
        # diagnosis & severity, plus follow-ups for the newly discovered ones.
        newly_unasked = [s for s in session_symptoms if s not in asked_followups_for]
        if not newly_unasked:
            # No brand-new symptoms => done for now
            break
        # Otherwise, we repeat to handle the newly discovered symptom(s).


def handle_faq_query(faq_model, user_input):
    faq_res = faq_model.get_best_match(user_input, top_k=1)
    if faq_res and faq_res[0]["score"] > 0.5:
        print(f"\nChatbot (FAQ): {faq_res[0]['answer']}")
    else:
        print("\nChatbot (FAQ): I'm not sure about that. Please try rephrasing or list symptoms.")


if __name__ == "__main__":
    main()
