# FakeNewsPredictionApp
Fake News Prediction App

                This web app is designed to predict whether a given news article is real or fake. It utilizes a logistic regression model trained on a dataset containing 20,000 sample news articles. The model achieves an impressive accuracy of 96%. To facilitate the prediction process, the app employs a TF-IDF vectorizer along with preprocessing techniques from the NLTK library, including lowering, regular expressions, splitting, and merging textual data.

                **Skills Enhanced:**

                :speech_balloon: Natural Language Processing (NLP)\n
                :computer: Machine Learning (ML)\n
                :snake: Python Programming\n
                :bar_chart: Data Manipulation and Analysis\n
                Steps Used to Create the Fake News Prediction App:

                **Step 1: Data Acquisition**\n
                The dataset for training the model was obtained from various sources, containing approximately 20,000 news articles. These articles span a range of topics and are labeled as either real or fake.

                **Step 2: Data Preprocessing**\n
                Before training the model, the textual data undergoes preprocessing steps to ensure compatibility with machine learning algorithms. This includes removing missing values, tokenization, converting text to lowercase, applying stemming to reduce word variations, and creating a unified "Tags" column containing processed text data.

                **Step 3: Data Visualization**\n
                To gain insights into the dataset and understand patterns, various visualization techniques are employed. This includes generating heatmaps, correlation matrices, and confusion matrices using Python libraries like Matplotlib.

                **Step 4: Model Creation**\n
                A logistic regression model is trained on the preprocessed dataset using a TF-IDF vectorizer. The TF-IDF vectorizer converts text data into numerical vectors, while logistic regression is utilized for classification. The model is capable of predicting whether a news article is real or fake with high accuracy.

                **Step 5: Evaluation**\n
                The model's performance is evaluated based on observation and analysis of its predictions. Both correct and incorrect predictions are noted to understand the strengths and weaknesses of the model. The high accuracy achieved indicates the effectiveness of the approach in distinguishing between real and fake news articles.

                Libraries Used:
    
                :fast_forward: numpy: For numerical operations\n
                :panda_face: pandas: For data manipulation\n
                :chart_with_upwards_trend: matplotlib.pyplot: For data visualization\n
                :book: nltk: For natural language processing tasks\n
                :arrows_counterclockwise: re: For regular expressions\n
                :gear: sklearn.feature_extraction.text.TfidfVectorizer: For converting text to vectors\n
                :scales: sklearn.model_selection.train_test_split: For splitting the data into training and testing sets\n
                :chart_with_downwards_trend: sklearn.linear_model.LogisticRegression: For logistic regression modeling\n
                :100: sklearn.metrics.accuracy_score: For calculating accuracy\n
                :mag: ydata_profiling.ProfileReport: For data profiling\n
                :pick: pickle: For saving and loading the trained model\n
                By leveraging NLP techniques and machine learning algorithms, this app serves as a valuable tool for identifying potentially misleading or false information in news articles, contributing to the fight against misinformation and promoting media literacy.
