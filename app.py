import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# App title
st.title("📧 Email Spam Classifier using NLP")

# Input box
input_msg = st.text_area("Enter your email message here:")

if st.button("Predict"):
    # Preprocess and predict
    transformed_msg = vectorizer.transform([input_msg])
    result = model.predict(transformed_msg)[0]

    if result == 1:
        st.error("❗ This email is SPAM")
    else:
        st.success("✅ This email is NOT SPAM")
