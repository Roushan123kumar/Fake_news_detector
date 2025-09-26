import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 

# Configure Streamlit to run on EB expected port
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = "8080"

st.set_page_config(page_title="Fake News Detector")

# Load trained model + vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# UI
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

inputn = st.text_area("News Article:", "")

if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success(" The News is Real!")
        else:
            st.error(" The News is Fake!")
    else:
        st.warning(" Please enter some text to Analyze.")
