import pandas as pd
import os
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')


#Please change your database location
# If automatically not working then please use manualy csv_file_path
# csv_file_path = '/home/hp/Practice/sample_data.csv'
current_dir = os.getcwd()

relative_path = 'sample_data.csv'
csv_file_path = os.path.join(current_dir, relative_path)
data = pd.read_csv(csv_file_path)

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('german'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    stemmer = SnowballStemmer('german')
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Apply text preprocessing to the 'text' column
data['text'] = data['text'].apply(preprocess_text)

# Check for and handle missing values in the training data
missing_values = data['text'].isnull().sum()
if missing_values > 0:
    data['text'] = data['text'].fillna('')

# Drop rows with missing labels
data = data.dropna(subset=['label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# Evaluate the model
predictions = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Save the model and vectorizer
# If automatically path is not working then please use manualy path
# model_file_path = '/home/hp/Practice/model.bin'
# vectorizer_file_path = '/home/hp/Practice/vectorizer.bin'

model_path = 'model.bin'
vectorizer_file_path = 'vectorizer.bin'

model_file_path = os.path.join(current_dir, model_path)
vectorizer_file_path = os.path.join(current_dir, vectorizer_file_path)
joblib.dump(model, model_file_path)
joblib.dump(vectorizer, vectorizer_file_path)

# Test phrases for classification
phrases = ["Das ist ein Test.", "wie gets", "Was ist die Hauptstadt von Deutschland?", "individuelle verpackung", "sandstrahlen von holz Lohn"]

# Preprocess the input phrases
preprocessed_phrases = [preprocess_text(phrase) for phrase in phrases]
input_tfidf = vectorizer.transform(preprocessed_phrases)

# Make predictions
predictions = model.predict(input_tfidf)
for phrase, prediction in zip(phrases, predictions):
    print(f"Phrase: {phrase} \nPredicted Label: {prediction}\n{'='*30}")
