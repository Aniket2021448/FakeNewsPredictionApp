import streamlit as st

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

st.set_page_config(page_title="News Prediction", page_icon=":earth_africa:")





# Load the model
news_model = pickle.load(open("fake_news_predictor_model.pkl", "rb"))
vectorizer = pickle.load(open("fakeNews_tfidf_vectorizer.pkl", "rb"))

# Function for preprocessing input text
def preProcessing(author, title, text):
    input_corpus = author +" " + title + " " + text
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
    
    
    # TO remove streamlit branding and other running animation
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Spinners
    bar = st.progress(0)
    for i in range(101):
        bar.progress(i)
        # time.sleep(0.02)  # Adjust the sleep time for the desired speed

    st.balloons()

    # Web content starts
    # Navbar starts
        # Create the Streamlit app
    col1, col2 = st.columns([1, 10])
    with col1:
        st.header("	:globe_with_meridians:")
    with col2:
        st.header("Fake News Prediction App")
        

    # Initialize NLTK resources
    nltk.download('stopwords')

    # Create sidebar section for app description and links
    st.sidebar.title("Find the fake :mag_right:")
    st.sidebar.write("Welcome the NLP based fake news detector :male-detective:")
    st.sidebar.write("""

                This web app predicts whether a given news article is real or fake using a logistic regression model trained on a dataset containing 20,000 sample news articles with an impressive accuracy of 96%. The app employs TF-IDF vectorization and NLTK library preprocessing techniques, including lowercase conversion, regular expressions, tokenization, stemming, and merging textual data.

                Skills Enhanced:

                💬 NLP
                💻 ML
                🐍 Python
                📊 Data Analysis
                
                    
\nSteps:   
                         
    1. Data Acquisition: Obtained a dataset of 20,000 news articles from various sources.\n
    2. Data Preprocessing: Handled missing values, tokenization, lowercase conversion, stemming, and unified text data.\n
    3. Data Visualization: Used Matplotlib for heatmaps, correlation, and confusion matrices.\n
    4. Model Creation: Trained a logistic regression model with TF-IDF vectorization for classification.\n
    5. Evaluation: Evaluated model performance with accuracy analysis.\n

By leveraging NLP and ML, this app helps identify false information in news articles, aiding in the fight against misinformation and promoting media literacy.
        
**Credits** 🌟\n
Coder: Aniket Panchal
GitHub: https://github.com/Aniket2021448

**Contact** 📧\n
For any inquiries or feedback, please contact aniketpanchal1257@gmail.com
    
    """)
    st.sidebar.write("Feel free to check out my other apps:")


    with st.sidebar.form("app_selection_form"):
        st.write("Feel free to explore my other apps :eyes:")
        app_links = {
            "Movie-mind": "https://movie-mind.streamlit.app/",
            "Comment-Feel": "https://huggingface.co/spaces/GoodML/Comment-Feel"
        }
        selected_app = st.selectbox("Choose an App", list(app_links.keys()))

        submitted_button = st.form_submit_button("Go to App")

    # Handle form submission
    if submitted_button:
        selected_app_url = app_links.get(selected_app)
        if selected_app_url:
            st.sidebar.success("Redirected successfully!")
            st.markdown(f'<meta http-equiv="refresh" content="0;URL={selected_app_url}">', unsafe_allow_html=True)

    
    # Dropdown menu for other app links

    st.sidebar.write("In case the apps are down, because of less usage")
    st.sidebar.write("Kindly reach out to me @ aniketpanchal1257@gmail.com")
    

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
        print("Prediction: ", prediction)
        print("Prediction[0]: ", prediction[0])
        if prediction[0] == 0:
            st.write("This news is predicted to be **real**.:muscle:")
        else:
            st.write("This news is predicted to be **fake**.:shit:")



if __name__ == "__main__":
    main()
