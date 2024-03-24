Fake News Prediction App

This web app is designed to predict whether a given news article is real or fake. It utilizes a logistic regression model trained on a dataset containing 20,000 sample news articles. The model achieves an impressive accuracy of 96%. To facilitate the prediction process, the app employs a TF-IDF vectorizer along with preprocessing techniques from the NLTK library, including lowering, regular expressions, splitting, and merging textual data.

Skills Enhanced:

💬 Natural Language Processing (NLP)

💻 Machine Learning (ML)

🐍 Python Programming

📊 Data Manipulation and Analysis

Steps Used to Create the Fake News Prediction App:

Step 1: Data Acquisition

The dataset for training the model was obtained from various sources, containing approximately 20,000 news articles. These articles span a range of topics and are labeled as either real or fake.

Step 2: Data Preprocessing

Before training the model, the textual data undergoes preprocessing steps to ensure compatibility with machine learning algorithms. This includes removing missing values, tokenization, converting text to lowercase, applying stemming to reduce word variations, and creating a unified "Tags" column containing processed text data.

Step 3: Data Visualization

To gain insights into the dataset and understand patterns, various visualization techniques are employed. This includes generating heatmaps, correlation matrices, and confusion matrices using Python libraries like Matplotlib.

Step 4: Model Creation

A logistic regression model is trained on the preprocessed dataset using a TF-IDF vectorizer. The TF-IDF vectorizer converts text data into numerical vectors, while logistic regression is utilized for classification. The model is capable of predicting whether a news article is real or fake with high accuracy.

Step 5: Evaluation

The model's performance is evaluated based on observation and analysis of its predictions. Both correct and incorrect predictions are noted to understand the strengths and weaknesses of the model. The high accuracy achieved indicates the effectiveness of the approach in distinguishing between real and fake news articles.

Libraries Used:

⏩ numpy: For numerical operations

🐼 pandas: For data manipulation

📈 matplotlib.pyplot: For data visualization

📖 nltk: For natural language processing tasks

🔄 re: For regular expressions

⚙️ sklearn.feature_extraction.text.TfidfVectorizer: For converting text to vectors

⚖️ sklearn.model_selection.train_test_split: For splitting the data into training and testing sets

📉 sklearn.linear_model.LogisticRegression: For logistic regression modeling

💯 sklearn.metrics.accuracy_score: For calculating accuracy

🔍 ydata_profiling.ProfileReport: For data profiling

⛏️ pickle: For saving and loading the trained model

By leveraging NLP techniques and machine learning algorithms, this app serves as a valuable tool for identifying potentially misleading or false information in news articles, contributing to the fight against misinformation and promoting media literacy.
