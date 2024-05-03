import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from text_processing import map_rating, preprocess_text

# Step 1: Load data from CSV file
df = pd.read_csv("datasets/tripadvisor_hotel_reviews.csv")
print("Data has been loaded from the CSV file.")

# Clean the data, removing any stop words
df['Review_Cleaned'] = df['Review'].apply(preprocess_text)
# Map ratings to categorical labels
df['Rating_Category'] = df['Rating'].apply(map_rating)

X = df['Review_Cleaned'].values
y = df['Rating_Category'].values

print("Data has been cleaned, and the Rating Category has been mapped.")

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print("Training data has been loaded.")

# Step 2 : Split the data into test and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print("Training data has been loaded.")


# Step 3: Convert text data into numerical vectors using TF-IDF as well as apply preprocessing
tfidf_vectorizer = TfidfVectorizer(stop_words='english',  # Remove English stopwords
                                   max_df=0.8,            # Ignore terms that appear in more than 80% of documents
                                   min_df=2,              # Ignore terms that appear in less than 2 documents
                                   max_features=1000)     # Limit the vocabulary size to top 1000 terms

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Data has been processed and loaded into numerical vectors.")


# Step 4: Train Multilayer Perceptron Classifier (Neural Network)
print("Neural Network training started.")
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
clf.fit(X_train_tfidf, y_train)
print("Neural Network has been trained.")

# Step 5: Evaluate neural network (Roughly 82%)
y_pred = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Neural Network Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Neural Network Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Neural Network Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("Neural Network F1-score:", f1)
