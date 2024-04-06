import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input

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

# Step 3: Preprocess text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max([len(seq) for seq in X_train_seq + X_test_seq])

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

# Step 4: Encode labels
unique_labels = list(set(y_train))
label_to_int = {label: i for i, label in enumerate(unique_labels)}

y_train_encoded = [label_to_int[label] for label in y_train]
y_test_encoded = [label_to_int[label] for label in y_test]

y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# Step 5: Define CNN architecture, follow their guide from that saved link
input_layer = Input(shape=(max_length,), dtype='int32')

model = Sequential([
    input_layer,
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # This is the convolution layer making this a CNN
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

# Step 6: Compile and train the model,
"""
0.2 Test Size

10 Epochs = 100% while training and 84% while testing accuracy
5 Epochs = 98% while training and 82% while testing accuracy


0.4 Test Size

5 Epochs = 99.7% while training and 82.4% while testing accuracy
"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train_encoded, validation_data=(X_test_padded, y_test_encoded), epochs=5, batch_size=64)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print("Test accuracy:", accuracy)
