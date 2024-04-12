import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text, case=0, remove_stops=True, stem=True):
    """
    Preprocess text by performing the following operations:
    - Convert text to the specified case (0 for lower, 1 for upper)
    - Remove stopwords (if remove_stops is True)
    - Stem words (if stem is True)

    Args:
    text (str): The input text to be preprocessed.
    case (int, optional): The case to convert the text to (0 for lower, 1 for upper). Default is 0 (lower).
    remove_stops (bool, optional): Whether to remove stopwords. Default is True.
    stem (bool, optional): Whether to perform stemming. Default is True.

    Returns:s
    str: The preprocessed text.
    """
    # Convert case
    if case == 0:
        text = text.lower()
    elif case == 1:
        text = text.upper()

    # Remove stopwords
    if remove_stops:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        text = ' '.join(filtered_words)

    # Perform stemming
    if stem:
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        text = ' '.join(stemmed_words)

    return text
