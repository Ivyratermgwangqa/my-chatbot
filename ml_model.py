import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import random
from preprocessing import preprocess  # Import preprocessing functions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Import knowledge base from main.py

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
current_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(current_dir, 'chatbot_data.csv'))

# Handle missing values
df['Query'] = df['Query'].fillna('')
df['Intent'] = df['Intent'].fillna('unknown')
df['Response'] = df['Response'].fillna('No response')

# Verify no missing values remain
df.info()

# Initialize NLTK utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Apply preprocessing to queries
df['Processed_Query'] = df['Query'].apply(preprocess)

# TF-IDF Vectorizer setup
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit(df['Processed_Query'])

knowledge_base = {
    "hello": ["Hi! How can I help you regarding Sol Plaatje University?", "Hello! What can I assist you with about SPU?", "Greetings! How may I assist you today regarding Sol Plaatje University?"],
    "hey": ["Hey there! How can I help you with information about Sol Plaatje University?", "Hey! What would you like to know about SPU?", "Hello! How may I assist you with SPU queries?"],
    "hi": ["Hi! How can I assist you with Sol Plaatje University today?", "Hello! What can I help you with regarding SPU?", "Greetings! How may I assist you with information on Sol Plaatje University?"],
    "greetings": ["Greetings! How can I assist you with Sol Plaatje University?", "Hello! What information can I provide about SPU?"],
    "courses": ["Sol Plaatje University offers various courses across faculties like Education, Humanities, Economic and Management Sciences, and Natural and Applied Sciences.", "SPU offers undergraduate and postgraduate programs. You can check the official website for details on specific courses."],
    "application_fee": ["No, you don't need to pay an application fee when applying for admission to Sol Plaatje University."],
    "faculties": ["Sol Plaatje University has four faculties: Education, Humanities, Economic and Management Sciences, and Natural and Applied Sciences.", "SPU has four faculties offering a range of programs. For detailed information, please visit the university's website."],
    "apply_admission": ["To apply for admission at Sol Plaatje University, you need to visit the official admissions portal and complete the online application form."],
    "accommodation": ["Yes, Sol Plaatje University does provide accommodation for students. For more details, please check the accommodation section on the university's website."],
    "apply_accommodation": ["To apply for accommodation at SPU, you should fill out the accommodation application form available on their website once you have received your admission offer."],
    "goodbye": ["Goodbye! Feel free to ask more questions about Sol Plaatje University anytime.", "Bye! Don't hesitate to reach out for more information about SPU.", "Take care! Let me know if you need further assistance with SPU."],
    "contact": ["You can contact Sol Plaatje University via their official website or call their administrative office @ +27 677 8765 for further information.", "For direct inquiries, visit SPU's contact page on their website."],
    "location": ["Sol Plaatje University is located in Kimberley, Northern Cape, South Africa."],
    "apply": ["You can apply to Sol Plaatje University through their online application portal on the official website.", "To apply for a course at SPU, visit their admissions portal and complete the online application form."],
    "establishment": ["Sol Plaatje University was established in 2014."],
    "library": ["Yes Sol Plaatje University has a library, it's library offers both physical and digital resources. It is open to students and provides access to research materials, e-books, and journals."],
    "name": ["My name is Gemmies!", "I am Gemmies i'm designed to assist you about SPU!"]
}

def rule_based_response(query):
    """Check for predefined responses from the knowledge base and return a single response."""
    query_lower = query.lower().strip()
    
    # Get the list of responses from the knowledge base
    responses = knowledge_base.get(query_lower)
    
    if responses:
        # Randomly select one response from the list
        return random.choice(responses)
    
    return None

def ml_get_response(query):
    """Get response using the machine learning model."""
    new_query_processed = [preprocess(query)]
    new_query_tfidf = vectorizer.transform(new_query_processed)
    existing_queries_tfidf = vectorizer.transform(df['Processed_Query'])
    
    # Compute cosine similarity between new and existing queries
    similarity_scores = cosine_similarity(new_query_tfidf, existing_queries_tfidf)
    index = similarity_scores.argmax()
    
    # Threshold for similarity to respond with default message if unclear
    if similarity_scores.max() < 0.1:
        return "I'm sorry, I don't understand that. You can ask about courses, admission, faculties, or accommodation at Sol Plaatje University."
    
    return df.iloc[index]['Response']

def get_response(query):
    """Get chatbot response based on query type."""
    if isinstance(query, list):
        query = query[0]  # Use first element if a list
    
    # Check for rule-based responses first
    simple_response = rule_based_response(query)
    if simple_response:
        return simple_response
    
    # If no rule-based match, use ML model
    return ml_get_response(query)

def get_chatbot_response(query):
    """Get response from the chatbot based on user input."""
    return get_response(query)  

# Stratified K-Fold Cross-Validation
def evaluate_model():
    accuracies = []
    
    # Remove low-frequency classes
    df_filtered = df[df['Intent'].isin(df['Intent'].value_counts()[df['Intent'].value_counts() > 1].index)]
    
    # Adjust n_splits for K-Fold
    n_splits = min(5, df_filtered['Intent'].value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = LogisticRegression()
    model_pipeline = make_pipeline(vectorizer, model)

    for train_index, test_index in skf.split(df_filtered['Query'], df_filtered['Intent']):
        X_train, X_test = df_filtered['Query'].iloc[train_index], df_filtered['Query'].iloc[test_index]
        y_train, y_test = df_filtered['Response'].iloc[train_index], df_filtered['Response'].iloc[test_index]

        # Fit the model
        model_pipeline.fit(X_train, y_train)

        correct_predictions = 0
        
        # Make predictions and calculate accuracy
        for test_query, expected_response in zip(X_test, y_test):
            predicted_response = get_response(test_query)
            
            if predicted_response == expected_response:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)

    # Output the average accuracy
    average_accuracy = np.mean(accuracies)
    print(f'Average accuracy: {average_accuracy * 100:.2f}%')

    return model_pipeline

# Save and load model functionality
def save_model(model_pipeline):
    joblib.dump(model_pipeline, os.path.join(current_dir, 'chatbot_model.pkl'))

def load_model():
    return joblib.load(os.path.join(current_dir, 'chatbot_model.pkl'))

if __name__ == "__main__":
    model_pipeline = evaluate_model()
# Uncomment below to train, evaluate, and save the model
# model_pipeline = evaluate_model()
# save_model(model_pipeline)