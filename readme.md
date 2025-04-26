# WellWise: An Intelligent Symptom Checker and Follow-Up Chatbot using NLP and Semantic Retrieval

This chatbot helps users:
- Analyze described symptoms
- Predict possible medical conditions
- Assess symptom severity
- Answer common health-related FAQs
- Ask follow-up questions for better symptom clarification.

---

## How to Run Locally

1. **Install dependencies**  
   Make sure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data (Optional for first-time setup)**
   If needed, you can preprocess datasets by running:
   ```bash
   python prepare_data.py
   python build_followup_from_parquet.py
   ```

3. **Launch the app**
   Run the following command to start the app locally:
   ```bash
   streamlit run app.py
   ```

## Features
- Symptom extraction from user sentences
- Intelligent disease prediction based on symptoms
- Severity classification (mild / moderate / severe)
- Follow-up health questions
- Medical FAQ answering