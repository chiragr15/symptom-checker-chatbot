import streamlit as st
from sympton_retrieval import SymptomRetrievalModel
import time

# Set page config
st.set_page_config(
    page_title="Medical Symptom Checker",
    page_icon="🏥",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    return SymptomRetrievalModel(cache_embeddings=False)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("🏥 Medical Symptom Checker Chatbot")
    st.write("""
    Welcome to the Medical Symptom Checker Chatbot! 
    Please describe your symptoms, and I'll help suggest possible conditions.
    """)
    
    # Load the model
    model = load_model()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get predictions from the model
        with st.chat_message("assistant"):
            with st.spinner("Analyzing symptoms..."):
                predictions = model.get_disease_predictions(prompt)
                
                if not predictions:
                    st.markdown("I'm not sure I understand. Could you please describe your symptoms in simple terms? For example: 'I have a fever and headache'")
                else:
                    st.markdown("Based on your symptoms, here are possible conditions:")
                    
                    for pred in predictions[:3]:  # Show top 3 predictions
                        # Color-coded confidence level
                        confidence_color = {
                            "Very High": "red",
                            "High": "orange",
                            "Moderate": "yellow",
                            "Low": "green"
                        }[pred['confidence_level']]
                        
                        st.markdown(f"""
                        **{pred['disease']}**
                        - Matched Symptom: {pred['matched_symptom']}
                        - Confidence: <span style='color:{confidence_color}'>{pred['confidence']}% - {pred['confidence_level']}</span>
                        """, unsafe_allow_html=True)
                    
                    # Add disclaimer
                    st.warning("""
                    ⚠️ **Disclaimer**: This is an AI-powered tool for informational purposes only. 
                    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
                    Always seek the advice of your physician or other qualified health provider with any questions you may have.
                    """)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Analyzed your symptoms and provided possible conditions."
        })

if __name__ == "__main__":
    main() 