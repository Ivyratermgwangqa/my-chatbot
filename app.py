from flask import Flask, render_template, request, jsonify
from main import get_response, load_ml_model  # Import from main.py
from preprocessing import preprocess  # Import preprocessing function

app = Flask(__name__)

# Load the ML model when the app starts
ml_model = load_ml_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input from the JSON request body
    user_input = request.json.get('message')

    if not user_input:  # Validate input
        return jsonify({'response': 'No input provided'}), 400

    # Preprocess the user input
    processed_input = preprocess(user_input)
    
    # Get the chatbot response using the knowledge base first, and ML if needed
    response = get_response(processed_input, model=ml_model)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)