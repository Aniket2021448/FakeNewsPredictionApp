import streamlit as st

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Initialize NLTK resources
nltk.download('stopwords')

# Load the model
news_model = pickle.load(open("fake_news_predictor_model.pkl", "rb"))
vectorizer = pickle.load(open("fakeNews_tfidf_vectorizer.pkl", "rb"))

# Function for preprocessing input text
def preProcessing(author, title, text):
    input_corpus = title + " " + author + " " + text
    input_corpus = re.sub('[^a-zA-Z]', ' ', input_corpus)
    input_corpus = input_corpus.lower()
    input_corpus = input_corpus.split()
    ps = PorterStemmer()
    input_corpus = [ps.stem(word) for word in input_corpus if not word in set(stopwords.words('english'))]
    input_corpus = ' '.join(input_corpus)
    return input_corpus

# Function to convert text into numerical vector using TF-IDF
def convertIntoVector(X):
    # Now converting the textual data into numerical vectors using the initialized TF-IDF vectorizer
    X = vectorizer.transform(X)
    return X

def main():
        

    # Create the form
    with st.form("news_form"):
        st.subheader("Enter News Details")
        author = st.text_input("Author Name")
        title = st.text_input("Title")
        text = st.text_area("Text")
        submit_button = st.form_submit_button("Submit")

    # Process form submission and make prediction
    if submit_button:

        input_text = preProcessing(author, title, text) 
        numerical_data = convertIntoVector([input_text])
        prediction = news_model.predict(numerical_data)
        
        st.subheader(":loudspeaker:Prediction:")
        if prediction[0] == 0:
            st.write("This news is predicted to be **real**.:muscle:")
        else:
            st.write("This news is predicted to be **fake**.:shit:")

