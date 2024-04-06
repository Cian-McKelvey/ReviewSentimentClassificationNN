import pandas as pd
import numpy as np
from keras import metrics  # Allows the reading of metrics while training
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input
from sklearn.model_selection import StratifiedKFold

from text_processing import custom_preprocessor


def map_rating(rating):
    if rating <= 2:
        return 'bad'
    elif rating == 3:
        return 'neutral'
    else:
        return 'good'


# Step 1: Load data from CSV file
df = pd.read_csv("datasets/tripadvisor_hotel_reviews.csv")
print("Data has been loaded from the CSV file.")

# Map ratings to categorical labels
df['Rating_Category'] = df['Rating'].apply(map_rating)

X = df['Review'].values  # numpy array of reviews
y = df['Rating_Category'].values  # numpy array of categorical labels

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print("Training data has been loaded.")

# Define the number of folds for cross-validation
n_folds = 5

# Define the K-fold cross-validator
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Define a list to store the evaluation results
cv_scores = []

fold_index = 0

# Iterate over the folds
for train_indices, val_indices in kfold.split(X_train, y_train):
    print(f"Begining fold: {fold_index}")
    fold_index += 1
    # Split data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
    y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

    # Step 3: Preprocess text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_fold)

    X_train_seq = tokenizer.texts_to_sequences(X_train_fold)
    X_val_seq = tokenizer.texts_to_sequences(X_val_fold)

    max_length = max([len(seq) for seq in X_train_seq + X_val_seq])

    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_length)

    # Step 4: Encode labels
    unique_labels = list(set(y_train_fold))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    y_train_encoded = [label_to_int[label] for label in y_train_fold]
    y_val_encoded = [label_to_int[label] for label in y_val_fold]

    y_train_encoded = to_categorical(y_train_encoded)
    y_val_encoded = to_categorical(y_val_encoded)

    # Step 5: Define CNN architecture
    input_layer = Input(shape=(max_length,), dtype='int32')

    model = Sequential([
        input_layer,
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])

    # Step 6: Compile and train the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.F1Score()])
    # Setting verbose to 1, or removing the arg altogether will show the progress, 0 will hide it
    model.fit(X_train_padded, y_train_encoded, validation_data=(X_val_padded, y_val_encoded), epochs=5, batch_size=64, verbose=1)

    # Evaluate the model on the validation set for this fold
    loss, accuracy = model.evaluate(X_val_padded, y_val_encoded, verbose=0)

    # Store the accuracy for this fold
    cv_scores.append(accuracy)

# Calculate the average accuracy across all folds
avg_accuracy = np.mean(cv_scores)
print("Average cross-validation accuracy:", avg_accuracy)

# Preprocess the test data using the tokenizer
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

# Encode the test labels
y_test_encoded = [label_to_int[label] for label in y_test]
y_test_encoded = to_categorical(y_test_encoded)

# Step 7: Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print("Test accuracy:", accuracy)

"""
Average cross-validation accuracy: 0.8139744997024536
257/257 ━━━━━━━━━━━━━━━━━━━━ 22s 84ms/step - accuracy: 0.8237 - loss: 0.8801
Test accuracy: 0.8181042075157166
"""
