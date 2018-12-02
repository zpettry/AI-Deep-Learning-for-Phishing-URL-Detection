#!/usr/bin/env python
"""
This file is for training on the PhishTank data.
"""

from __future__ import print_function
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, Dense, Flatten, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np
import label_data

# Get and process URL data and labels.
urls = label_data.main()

samples = []
labels = []
for k, v in urls.items():
    samples.append(k)
    labels.append(v)
    #print(k, v)
    
print(labels.count(1))
print(labels.count(0))

# Preprocess data for training.
max_chars = 20000
maxlen = 128

tokenizer = Tokenizer(num_words=max_chars, char_level=True)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Divide data between training, cross-validation, and test data.
training_samples = int(len(samples) * 0.95)
validation_samples = int(len(labels) * 0.05)
print(training_samples, validation_samples)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
'''
x = data
y = labels
'''
x = data[:training_samples]
y = labels[:training_samples]
x_test = data[training_samples: training_samples + validation_samples]
y_test = labels[training_samples: training_samples + validation_samples]

# Define callbacks for Keras.
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
    filepath='lstmchar256256128test.h5',
    monitor='val_loss',
    save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0,
    patience=2, 
    mode='auto',
    baseline=None,
    )
]

num_chars = len(tokenizer.word_index)+1

embedding_vector_length = 128

# Create model for training.
model = Sequential()
model.add(Embedding(num_chars, embedding_vector_length, input_length=maxlen))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Train.
model.fit(x, y,
        epochs=10,
        batch_size=1200,
        callbacks=callbacks_list,
        validation_split=0.20,
        shuffle=True
        )

# Evaluate model on test data.
score, acc = model.evaluate(x_test, y_test, verbose=1, batch_size=1024)

print("Model Accuracy: {:0.2f}%".format(acc * 100))

