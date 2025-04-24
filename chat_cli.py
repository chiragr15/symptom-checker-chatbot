from sympton_retrieval import SymptomRetrievalModel
from symptom_severity_checker import SymptomSeverityChecker
from followup import get_followup_questions
from symptom_utils import extract_symptoms_from_sentence
from faq_chatbot import FAQChatbot


def main():
    retriever = SymptomRetrievalModel(cache_embeddings=False)
    severity_checker = SymptomSeverityChecker()
    faq_model = FAQChatbot("data/faq_dataset.csv")

    print("Symptom Checker Chatbot (CLI Mode)")
    print("Type 'exit' to quit\n")

    session_symptoms = set()
    asked_followups_for = set()

    ambiguous_input = ""

    while True:
        if ambiguous_input:
            user_input = ambiguous_input
            ambiguous_input = ""
        else:
            user_input = input(" Enter symptoms or ask a question: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            break

        # (A) Detect new symptoms
        new_symptoms = set(extract_symptoms_from_sentence(user_input, retriever.symptom_vocab_list))

        if new_symptoms:
            session_symptoms |= new_symptoms
            print(f"\n [User Provided New Symptoms] => {new_symptoms}")
            print(f" Current All Symptoms: {session_symptoms}")

            ambiguous_input = run_diagnosis_and_followups(
                session_symptoms,
                asked_followups_for,
                retriever,
                severity_checker,
                faq_model
            )
            continue

        # (B) No new symptoms found → clarify intent
        user_question = user_input.lower()
        is_question = (
            "?" in user_question or
            user_question.startswith(("what", "how", "can", "should", "is", "do", "does", "will", "could"))
        )

        if is_question:
            handle_faq_query(faq_model, user_input)
            continue

        print("\n I didn’t detect any new symptoms.")
        print("Would you like to:")
        print("  1.  Add a symptom")
        print("  2.  Ask a health question")
        print("  3.  Continue with current symptom list")

        followup_choice = input("\n Please type: add / question / continue: ").strip().lower()

        if followup_choice.startswith("add"):
            print("\nYou can now enter the symptom you'd like to add.")
            continue
        elif followup_choice.startswith("question"):
            handle_faq_query(faq_model, user_input)
            continue
        else:
            print("\n Continuing with current symptoms...")
            ambiguous_input = run_diagnosis_and_followups(
                session_symptoms,
                asked_followups_for,
                retriever,
                severity_checker,
                faq_model
            )
            continue


def run_diagnosis_and_followups(
    session_symptoms: set,
    asked_followups_for: set,
    retriever,
    severity_checker,
    faq_model
) -> str:
    while True:
        symptom_list_str = ", ".join(session_symptoms)
        disease_results = retriever.get_disease_predictions(symptom_list_str)
        if not disease_results:
            print("\n No disease predictions found.")
        else:
            print("\n Current Predicted Conditions (based on all known symptoms):")
            for res in disease_results:
                print(
                    f" - {res['disease']} (matched with '{res['matched_symptom']}') "
                    f"[confidence: {res['confidence']}% - {res['confidence_level']}]"
                )

        if session_symptoms:
            severity_results = severity_checker.classify_severity(symptom_list_str)
            print("\n Severity Assessment:")
            for sres in severity_results:
                print(
                    f" - {sres['symptom']}: Severity={sres['severity'].capitalize()} → {sres['alert']}"
                )
        else:
            print("\n No known symptoms to assess severity.")

        unasked_symptoms = [s for s in session_symptoms if s not in asked_followups_for]
        if not unasked_symptoms:
            break

        for symptom in unasked_symptoms:
            followups = get_followup_questions(symptom)
            asked_followups_for.add(symptom)

            if not followups:
                continue

            print(f"\nFollow-up questions for the newly mentioned symptom '{symptom}':")
            for i, q in enumerate(followups, 1):
                print(f"  {i}. {q}")

            user_answer = input("\n Your answer to the above follow-up questions or please ask any other question you have: ").strip()
            if user_answer.lower() in ["exit", "quit"]:
                exit(0)

            newly_found = set(extract_symptoms_from_sentence(user_answer, retriever.symptom_vocab_list))
            if newly_found:
                print(f"  [New Follow-up Symptoms] => {newly_found}")
                session_symptoms |= newly_found
                print(f" Current All Symptoms: {session_symptoms}")
            else:
                # Return this input to main() for clarification
                return user_answer

        newly_unasked = [s for s in session_symptoms if s not in asked_followups_for]
        if not newly_unasked:
            break

    return ""  # No ambiguous input to return


def handle_faq_query(faq_model, user_input):
    faq_res = faq_model.get_best_match(user_input, top_k=1)
    if faq_res and faq_res[0]["score"] > 0.5:
        print(f"\nChatbot (FAQ): {faq_res[0]['answer']}")
    else:
        print("\nChatbot (FAQ): I’m not sure how to answer that.")
        print("But here’s what I know based on your symptoms so far.")


if __name__ == "__main__":
    main()
