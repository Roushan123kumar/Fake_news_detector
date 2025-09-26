import streamlit as st
import joblib
import boto3
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit page config
st.set_page_config(page_title="Fake News Detector")

# AWS S3 credentials
AWS_ACCESS_KEY = "AKIAVHXSPQR3P3JIMJWC"
AWS_SECRET_KEY = "WzukeJGMe993Wxi9Oyik/KzI83lUZxgw4Qgog8B0"
BUCKET_NAME = "your-bucket-name"  # <-- Replace with your actual bucket name

# Function to load files from S3
def load_from_s3(file_name):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_name)
    return joblib.load(BytesIO(obj['Body'].read()))

# Load vectorizer and model from S3
vectorizer = load_from_s3("vectorizer.jb")
model = load_from_s3("lr_model.jb")

# Streamlit UI
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
        st.warning(" Please enter some text to analyze.")
