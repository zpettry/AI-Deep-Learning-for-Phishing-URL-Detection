#!/usr/bin/env python
"""
This is the Flask REST API that processes and outputs the prediction on the URL.
"""
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import label_data
import flask
import json

# Initialize our Flask application and the Keras model.
app = flask.Flask(__name__)

global graph
graph = tf.get_default_graph()
model_pre = 'bi-lstmchar256256128.h5'
model = load_model(model_pre)

def prepare_url(url):

    urlz = label_data.main()

    samples = []
    labels = []
    for k, v in urlz.items():
        samples.append(k)
        labels.append(v)

    #print(len(samples))
    #print(len(labels))

    maxlen = 128
    max_words = 20000

    tokenizer = Tokenizer(num_words=max_words, char_level=True)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(url)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))

    url_prepped = pad_sequences(sequences, maxlen=maxlen)
    return url_prepped

@app.route("/predict", methods=["POST"])
def predict():

    # Initialize the dictionary for the response.
    data = {"success": False}

    # Check if POST request.
    if flask.request.method == "POST":
		
        # Grab and process the incoming json.
        incoming = flask.request.get_json()
        urlz = []
        url = incoming["url"]

        urlz.append(url)
        print(url)

        # Process and prepare the URL.
        url_prepped = prepare_url(urlz)

        # classify the URL and make the prediction.
        with graph.as_default():
            prediction = model.predict(url_prepped)
        print(prediction)
        
        data["predictions"] = []
        
        if prediction > 0.50:
            result = "URL is probably malicious."
        else:
            result = "URL is probably NOT malicious."
        
	# Check for base URL. Accuracy is not as great.
        split = url.split("//")
        print(split[0])
        split2 = split[1]
        if "/" not in split2:
            result = "Base URLs cannot be accurately determined."
        
	# Processes prediction probability.
        prediction = float(prediction)
        prediction = prediction * 100
        
        if result == "Base URLs cannot be accurately determined.":
            r = {"result": result, "url": url}
        else:
            r = {"result": result, "malicious percentage": prediction, "url": url}
        data["predictions"].append(r)

        # Show that the request was a success.
        data["success"] = True

    # Return the data as a JSON response.
    return flask.jsonify(data)

# Start the server.
if __name__ == "__main__":
    print("Starting the server and loading the model...")
    app.run(host='0.0.0.0', port=45000)

