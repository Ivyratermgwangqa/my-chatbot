import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import string
import joblib
import os
from ml_model import load_model, get_response as ml_get_response
from ml_model import knowledge_base
from preprocessing import preprocess, get_synonyms

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 

def extract_entities(tokens):
    entities = [token for token in tokens if token in knowledge_base]
    return entities

def pattern_matching(user_input):
    """
    Matches user input against pre-defined patterns to identify intent.
    Returns the matched intent if found, else returns None.
    """
    patterns = {
        "application_fee": r"\b(application fee|is there an application fee|application costs?|any fees to apply?)\b",
        "faculties": r"\b(how many faculties does spu have|number of faculties|faculties|what faculties are offered)\b",
        "apply_admission": r"\b(how do I apply for admission|how to apply for admission|apply for admission|application process)\b",
        "accommodation": r"\b(does it have accommodation|accommodation|housing|residence|res|is housing available)\b",
        "apply_accommodation": r"\b(how do I apply for accommodation|apply for accommodation|apply for residence|apply for res|housing application)\b",
        "courses": r"\bcourses?\b|\bsubjects?\b|what\s*courses\b|course\s*offerings?",
        "library": r"\blibrary\b|\bresources\b|library\s*information",
        "location": r"\blocation\b|\blocated\b|\bwhere\s*is\b|find\s*us",
        "contact": r"\bcontact\b|\bget\s*in\s*touch\b|reach\s*out",
        "apply": r"\bapply\b|\bapplication\b|how\s*to\s*apply",
        "hello": r"\b(hi|hello|hey|greetings)\b",
        "goodbye": r"\b(goodbye|bye|exit|quit)\b",
        "establishment": r"\b(when was spu established|establishment|founding year|established)\b",
        "name": r"\b(what is your name|what's your name|who are you|name)\b",
    }

    for key, pattern in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return key
    return None

def get_response(user_input, model=None):
    """
    Generate a response based on user input by using rule-based methods or an ML model if provided.
    This function also handles named entity extraction, synonym mapping, and preprocessing.
    
    Args:
    user_input (str): The user's query or message.
    model (optional): The machine learning model for intent recognition and response generation.

    Returns:
    str: The chatbot's response.
    """

    normalized_input = user_input.lower().replace("sol plaatje university", "spu").replace("spu", "sol plaatje university")
    
    processed_input = preprocess(normalized_input)
    
    synonyms = get_synonyms(processed_input)
    
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    elif synonyms is None:
        synonyms = []

    expanded_input = [processed_input] + synonyms
    
    entities = extract_entities(processed_input)
    if entities:
        print("Extracted Entities:", entities)
    
    for query_variant in expanded_input:
        matched_entity = pattern_matching(query_variant)
        if matched_entity:
            return knowledge_base.get(matched_entity, ["I'm sorry, I don't understand that."])[0]
    
    if model:
        return ml_get_response(user_input)
    
    return "I'm sorry, I don't understand that. You can ask about courses, admission, faculties, or accommodation at Sol Plaatje University."


def load_ml_model():
    """Loads the trained machine learning model."""
    return load_model()

def chatbot():
    print("Chatbot: Hello! How can I assist you with university information today?")
    model = load_ml_model()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input, model)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()