import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# If this is first time running program, then uncomment to download needed resources
"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""



def email_reduction(file_path):
    with open(file_path, "r") as file:
        email_text = file.read()
    email_text = email_text.lower()
    tokens = word_tokenize(email_text)

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove common words like 'And' and 'The'
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Group words to condense them. (ex: Running -> Run)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join all the tokens together into one string
    preprocessed_email = " ".join(lemmatized_tokens)
    return preprocessed_email

