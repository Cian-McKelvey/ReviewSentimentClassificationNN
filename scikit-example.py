import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from text_processing import custom_preprocessor


# This method will set a rating of two or less to bad, etc...
def map_rating(rating):
    if rating <= 2:
        return 'bad'
    elif rating == 3:
        return 'neutral'
    else:
        return 'good'


# THis can maybe be used as one of the examples, and then the CNN can be the main example


# Step 1 : Create the datasets from the csv files
df = pd.read_csv("datasets/tripadvisor_hotel_reviews.csv")
print("Data has been loaded from the CSV file.")

# This gets the AI to try return their rating (roughly 55% accuracy)
# X = df['Review'].values  # numpy array of reviews
# y = df['Rating'].values  # numpy array of ratings

# OR

# This tries to map a review to a good, bad, neutral rating (roughly 80% accuracy)
df['Rating_Category'] = df['Rating'].apply(map_rating)

X = df['Review'].values  # numpy array of reviews
y = df['Rating_Category'].values  # numpy array of categorical labels


# Step 2 : Split the data into test and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print("Training data has been loaded.")


# Step 3: Convert text data into numerical vectors using TF-IDF as well as apply preprocessing
tfidf_vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor,  # Use the custom text preprocessor
                                   stop_words='english',  # Remove English stopwords
                                   max_df=0.8,            # Ignore terms that appear in more than 80% of documents
                                   min_df=2,              # Ignore terms that appear in less than 2 documents
                                   max_features=1000)     # Limit the vocabulary size to top 1000 terms

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Data has been processed and loaded into numerical vectors.")


# Step 4: Train Multinomial Naive Bayes classifier
print("Naive Bayes model training started.")
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)
print("Naive Bayes Model has been trained.")


# Step 5: Evaluate classifier
accuracy = clf.score(X_test_tfidf, y_test)
print("Bayes Model Accuracy:", accuracy)


# Step 6: Train Multilayer Perceptron Classifier (Neural Network)
print("Neural Network training started.")
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
clf.fit(X_train_tfidf, y_train)
print("Neural Network has been trained.")

# Step 5: Evaluate neural network (Roughly 82%)
accuracy = clf.score(X_test_tfidf, y_test)
print("Neural Network Accuracy:", accuracy)
