import pandas as pd
import numpy as np

from keras import metrics
from keras.src.layers import GlobalMaxPooling1D
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout

import matplotlib.pyplot as plt
from keras.regularizers import l2


from text_processing import preprocess_text, map_rating

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

# Step 3: Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max([len(seq) for seq in X_train_seq + X_test_seq])

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

print("Data has been tokenized and padded.")

# Step 4: Encode labels
unique_labels = list(set(y_train))
label_to_int = {label: i for i, label in enumerate(unique_labels)}

y_train_encoded = [label_to_int[label] for label in y_train]
y_test_encoded = [label_to_int[label] for label in y_test]

y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# Step 5: Define CNN architecture
input_layer = Input(shape=(max_length,), dtype='int32')
# Fourth attempt - 0.8261559009552002
model = Sequential([
    # Embedding layer to convert input text to dense vectors of fixed size
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),

    # Convolutional layer to extract features from the embedded representations of the text
    # L2 regularization added to reduce overfitting
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),

    # Global max pooling layer to reduce dimensionality and extract the most important features
    GlobalMaxPooling1D(),

    # Dense layer with ReLU activation to learn complex patterns from the extracted features
    # L2 regularization added to reduce overfitting
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),

    # Dropout layer to reduce overfitting by randomly dropping units during training
    Dropout(0.5),

    # Output layer with softmax activation for multi-class classification
    Dense(len(unique_labels), activation='softmax')
])

# Step 6: Compile and train the model,
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.Precision(),
                                                                          metrics.Recall(), metrics.F1Score()])

history = model.fit(X_train_padded, y_train_encoded, validation_data=(X_test_padded, y_test_encoded),
                    epochs=10, batch_size=64)


# Step 7: Evaluate the model - With all the new metrics added
results = model.evaluate(X_test_padded, y_test_encoded)
loss = results[0]
accuracy = results[1]

# Get predicted labels for the test set
y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision
precision = precision_score(np.argmax(y_test_encoded, axis=1), y_pred_classes, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(np.argmax(y_test_encoded, axis=1), y_pred_classes, average='weighted')
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(np.argmax(y_test_encoded, axis=1), y_pred_classes, average='weighted')
print("F1-score:", f1)

print("Test loss:", loss)
print("Test accuracy:", accuracy)
print("\n\n\n\n\n\n\n\n\n")
model.summary()


# Step 8: Plot the graph
# Plot the training and validation metrics
plt.figure(figsize=(12, 6))

# Plot the accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
