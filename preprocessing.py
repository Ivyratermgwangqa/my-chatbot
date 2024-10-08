import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_synonyms(word):
    """
    Fetches the most common synonym for a given word using WordNet.
    If no synonym is found, returns the original word.
    """
    synonyms = []
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    
    # Return the first synonym found, or the original word if no synonym is available
    return synonyms[0] if synonyms else word

def preprocess(text):
    """
    Preprocesses the input text by removing non-alphabetic characters, tokenizing, removing stopwords,
    applying lemmatization, and replacing words with their synonyms.
    """
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization and synonym replacement
    processed_tokens = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        synonym = get_synonyms(lemma)  # Replace with synonym
        processed_tokens.append(synonym)

    return ' '.join(processed_tokens)

# Example usage
sample_text = "What accommodation options are available?"
print(preprocess(sample_text))