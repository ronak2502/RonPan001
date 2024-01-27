# Text Classification with Machine Learning and REST API

This project includes a machine learning model for text classification and a Flask-based REST API to serve predictions. The model predicts the labels of input phrases based on its training.

## Machine Learning Model (`task.py`)

### Overview:

- **Data Setup:**
  - Reads data from a CSV file named `sample_data.csv` that contains text and corresponding labels.
  - Cleans and preprocesses the text data.

- **Training and Testing:**
  - Splits the dataset into training and testing sets.

- **Text Processing:**
  - Uses TF-IDF vectorization to convert text into numerical features.

- **Model Building:**
  - Chooses a linear Support Vector Machine (SVM) for classification.
  - Trains the model on the training data.

- **Evaluation:**
  - Assesses the model's performance on the testing set, providing accuracy and a classification report.

- **Model Saving:**
  - Saves the trained model (`model.bin`) and TF-IDF vectorizer (`vectorizer.bin`) for later use.

- **Testing Phrases:**
  - Includes sample phrases for testing.
  - Processes and predicts their labels using the trained model.
  - Prints the predicted labels.

## REST API (`app.py`)

### Overview:

- **Dependencies and Setup:**
  - Loads necessary libraries and sets up the Flask application.

- **Model Loading:**
  - Loads the pre-trained SVM model and TF-IDF vectorizer.

- **Text Processing:**
  - Defines a function to preprocess text.

- **Prediction Endpoint:**
  - Establishes an endpoint `/predict` that accepts POST requests with JSON data.
  - Processes incoming phrases, preprocesses them, and predicts their labels using the loaded model.
  - Returns the predicted labels in JSON format.

- **Error Handling:**
  - Implements basic error handling to log errors and return a server error message if something goes wrong.

## Dockerfile

### Overview:

- **Base Image:**
  - Utilizes Python 3.8 as the base image.

- **Working Directory and Copying Files:**
  - Sets the working directory inside the container to `/app`.
  - Copies the entire project directory into the container.

- **Package Installation:**
  - Installs required Python packages specified in `requirements.txt`.

- **Exposed Port:**
  - Exposes port 80 for external communication.

- **Entry Point:**
  - Configures the entry point to run a Bash script named `run.sh`.


## Follow these steps to run the model on your local machine.

## Clone the Repository
- git clone https://github.com/ronak2502/RonPan001.git
- cd RonPan001

- Please add `sample_data.csv` in project folder

## Build the Docker image
- docker build -t machine_lerning_model .

## Run the Docker container
- docker run -p 5000:5000 machine_learning_model

## Test the API
- curl -X POST -H "Content-Type: application/json" -d '{"phrases": ["Das ist ein Test.", "wie gehts", "Was ist die Hauptstadt von Deutschland?", "individuelle Verpackung", "sandstrahlen von Holz Lohn"]}' http://127.0.0.1:5000/predict



### Manual Setup: If you encounter any issues with Docker then please run the model and REST API manually. 

## Clone the Repository
- git clone https://github.com/ronak2502/RonPan001.git
- cd RonPan001

- Please add `sample_data.csv` in project folder

## Install Dependencies
- pip install -r requirements.txt

## Run task.py file to train the machine learning model
- python3 task.py

## Run the REST API
- python3 app.py

## Test the API
- curl -X POST -H "Content-Type: application/json" -d '{"phrases": ["Das ist ein Test.", "wie gehts", "Was ist die Hauptstadt von Deutschland?", "individuelle Verpackung", "sandstrahlen von Holz Lohn"]}' http://127.0.0.1:5000/predict
