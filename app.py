import streamlit as st
from chat_cli import (
    SymptomRetrievalModel,
    SymptomSeverityChecker,
    FAQChatbot,
    extract_symptoms_from_sentence,
    get_followup_questions
)
import io
import sys

# Set page config
st.set_page_config(
    page_title="Medical Symptom Checker",
    page_icon="🏥",
    layout="wide"
)

# Initialize models
@st.cache_resource
def load_models():
    return {
        #'retriever': SymptomRetrievalModel(cache_embeddings=False),
        'severity_checker': SymptomSeverityChecker(),
        'faq_model': FAQChatbot("data/faq_dataset.csv")
    }

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_symptoms' not in st.session_state:
    st.session_state.session_symptoms = set()
if 'asked_followups_for' not in st.session_state:
    st.session_state.asked_followups_for = set()

def capture_output(func, *args, **kwargs):
    """Capture print output from a function"""
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        func(*args, **kwargs)
        output = new_stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    return output

def process_user_input(prompt, models):
    # Extract new symptoms
    new_symptoms = set(extract_symptoms_from_sentence(prompt, models['retriever'].symptom_vocab_list))
    
    if new_symptoms:
        # Add new symptoms to session
        st.session_state.session_symptoms.update(new_symptoms)
        st.markdown(f"I've noted these new symptoms: {', '.join(new_symptoms)}")
        
        # Get disease predictions
        symptom_list_str = ", ".join(st.session_state.session_symptoms)
        disease_results = models['retriever'].get_disease_predictions(symptom_list_str)
        
        if disease_results:
            st.markdown("**Current Predicted Conditions:**")
            for res in disease_results:
                confidence_color = {
                    "Very High": "red",
                    "High": "orange",
                    "Moderate": "yellow",
                    "Low": "green"
                }[res['confidence_level']]
                
                st.markdown(f"""
                • **{res['disease']}**
                  - Matched with: {res['matched_symptom']}
                  - Confidence: <span style='color:{confidence_color}'>{res['confidence']}% ({res['confidence_level']})</span>
                """, unsafe_allow_html=True)
        
        # Get severity assessment
        if st.session_state.session_symptoms:
            severity_results = models['severity_checker'].classify_severity(symptom_list_str)
            st.markdown("**Severity Assessment:**")
            for sres in severity_results:
                st.markdown(f"""
                • {sres['symptom']}: {sres['severity'].capitalize()}
                  → {sres['alert']}
                """)
        
        # Get follow-up questions for new symptoms
        for symptom in new_symptoms:
            if symptom not in st.session_state.asked_followups_for:
                followups = get_followup_questions(symptom)
                if followups:
                    st.markdown(f"**Follow-up questions for '{symptom}':**")
                    for i, q in enumerate(followups, 1):
                        st.markdown(f"{i}. {q}")
                st.session_state.asked_followups_for.add(symptom)
    else:
        # Handle as FAQ
        faq_res = models['faq_model'].get_best_match(prompt, top_k=1)
        if faq_res and faq_res[0]["score"] > 0.5:
            st.markdown(faq_res[0]["answer"])
        else:
            st.markdown("I'm not sure about that. Could you please describe your symptoms or rephrase your question?")
    
    # Add medical disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer**: This is an AI-powered tool for informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider.
    """)

def main():
    st.title("🏥 Medical Symptom Checker Chatbot")
    st.write("""
    Welcome! Please describe your symptoms or ask any health-related questions.
    I'll help assess your symptoms and suggest possible conditions.
    """)
    
    # Load models
    models = load_models()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms or ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process input using chat_cli's functionality
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                process_user_input(prompt, models)
    
    # Show current symptoms in sidebar
    with st.sidebar:
        st.subheader("Current Symptoms")
        if st.session_state.session_symptoms:
            for symptom in st.session_state.session_symptoms:
                st.write(f"• {symptom}")
            if st.button("Clear Symptoms"):
                st.session_state.session_symptoms.clear()
                st.session_state.asked_followups_for.clear()
                st.rerun()
        else:
            st.write("No symptoms recorded yet.")

if __name__ == "__main__":
    main() 