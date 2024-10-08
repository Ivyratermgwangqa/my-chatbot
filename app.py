from flask import Flask, render_template, request, jsonify
from main import get_response, load_ml_model
from preprocessing import preprocess
app = Flask(__name__)

ml_model = load_ml_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'response': 'No input provided'}), 400

    processed_input = preprocess(user_input)
    
    response = get_response(processed_input, model=ml_model)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)