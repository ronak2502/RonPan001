from flask import Flask, request, jsonify
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


app = Flask(__name__)

# Load the trained model and vectorizer
# model_file_path = '/home/hp/Practice/model.bin'
# vectorizer_file_path = '/home/hp/Practice/vectorizer.bin'

current_dir = os.getcwd()
model_path = 'model.bin'
vectorizer_file_path = 'vectorizer.bin'

model_file_path = os.path.join(current_dir, model_path)
vectorizer_file_path = os.path.join(current_dir, vectorizer_file_path)


model = joblib.load(model_file_path)
vectorizer = joblib.load(vectorizer_file_path)

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('german'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    stemmer = SnowballStemmer('german')
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    phrases = data['phrases']
    
    # Preprocess the input phrases
    preprocessed_phrases = [preprocess_text(phrase) for phrase in phrases]

    # Convert text data to TF-IDF vectors
    input_tfidf = vectorizer.transform(preprocessed_phrases)
    predictions = model.predict(input_tfidf)

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=5000)
