# Define a custom preprocessor function
def custom_preprocessor(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    return text
