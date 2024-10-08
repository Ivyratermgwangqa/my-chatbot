# Chatbot Project

Welcome to the **Chatbot Project**! This repository contains a Flask application for a chatbot that integrates both machine learning and rule-based responses to assist users with various queries. The project also includes a Jupyter Notebook for running the chatbot without a user interface.

## Project Structure

The project is organized as follows:

```
my-chatbot/
│
├── app.py                     # Main Flask application
├── ml_model.py                # Machine learning model for the chatbot
├── main.py                    # Main entry point for additional functionalities
├── preprocessing.py           # NLP preprocessing functions
├── chatbot.ipynb              # Jupyter Notebook for chatbot (without UI)
│
├── static/
│   └── style.css              # CSS styles for the chatbot UI
│
├── templates/
│   └── index.html             # HTML for the chatbot interface
│
├── chatbot_data.csv           # CSV dataset for training the ML model
│
└── requirements.txt           # Python package dependencies
```

### Code Modules

#### **1.1. ml_model.py**

This module handles the machine learning aspects of the chatbot, such as training and prediction for intent recognition.

#### **1.2. preprocessing.py**

This module handles the natural language processing (NLP) preprocessing tasks, including tokenization, stop word removal, and other preprocessing techniques.

#### **1.3. app.py**

The Flask application that serves the chatbot. It handles the user input, processes it, and generates appropriate responses using both rule-based and machine learning approaches.

#### **1.4. chatbot.ipynb**

A Jupyter Notebook that allows you to run and test the chatbot without a user interface. Useful for debugging and experimenting with the chatbot's functionality.

### Additional Files

- **static/style.css**: Contains CSS styles for the chatbot UI.
- **templates/index.html**: HTML file for the chatbot interface.
- **chatbot_data.csv**: The dataset used for training the machine learning model.
- **requirements.txt**: Lists the Python dependencies required to run the chatbot.

## Requirements

Before running the application, ensure that you have the following installed:

- Python 3.x
- Flask
- scikit-learn
- pandas
- nltk
- joblib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ivyratermgwangqa/my-chatbot.git
   cd my-chatbot
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK resources:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Running the Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Interact with the chatbot through the web interface!

## Running the Chatbot Without UI

If you want to run the chatbot without the web interface, you can use the Jupyter Notebook:

1. Start the Jupyter Notebook:

   ```bash
   jupyter notebook chatbot.ipynb
   ```

2. Follow the steps in the notebook to interact with the chatbot directly.

## Contributing

We welcome contributions to this project! If you find any issues or have suggestions for improvements, feel free to create issues or submit pull requests. Your feedback and contributions are highly appreciated.

## Authors

- **Lerato Mgwangqa** - [GitHub](https://github.com/Ivyratermgwangqa)
- **Yinhla Chauke** - [GitHub](https://github.com/Yinhla-Chauke)

---

### 3. requirements.txt

Here's a sample `requirements.txt` that includes all the necessary libraries for the chatbot:

```
Flask==2.2.2
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.6.2
joblib==1.2.0
nltk==3.8.1
regex==2023.3.23
```

### Final Steps

- Ensure that all your Python scripts are functioning correctly and are properly documented.
- Test the application to ensure everything works as expected, both in the web interface and in the Jupyter Notebook.
