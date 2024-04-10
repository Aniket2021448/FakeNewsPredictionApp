import streamlit as st


def main():
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
        st.write("Select an App:")
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
    
    st.title("CONTACT ME")
    st.markdown("Github: https://github.com/Aniket2021448", unsafe_allow_html=True)
    st.markdown("Linked IN: https://www.linkedin.com/in/aniket-panchal-0a7b3a233/", unsafe_allow_html=True)
    st.markdown("Email: aniketpanchal1257@gmail.com", unsafe_allow_html=True)

    st.markdown("Github repository: https://github.com/Aniket2021448/FakeNewsPredictionApp", unsafe_allow_html=True)
    st.write("THANK YOU FOR CONNECTING")
