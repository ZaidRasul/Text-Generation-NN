import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np

# get the text file for training.
filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #open the file , read it, decode it to utf-8 and convert to lower case 

# taking sample of text for training
text = text[50000:150000]
characters = sorted(set(text)) # get the unique characters in the text and sort them

# create a dictionary that maps each character to an index and back
char_to_index = dict((char, index) for index, char in enumerate(characters))
index_to_char = dict((index, char) for index, char in enumerate(characters))

seq_length = 40 # length of each sequence to consider for next character prediction
step = 3 # step size to move forward in the text to create the next sequence
sentences = [] # list to hold the sequences of characters
next_char = [] # list to hold the next character for each sequence

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i+seq_length])
    next_char.append(text[i + seq_length])

# set true for a position if a character is present at that position in the sequence against a sentence
x = np.zeros(((len(sentences), seq_length, len(characters))), dtype=np.bool) # input data

y = np.zeros((len(sentences), len(characters)), dtype=np.bool) # output data
#to fill the arrays
for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1


# build the model
model = Sequential()
model.add(LSTM(128, input_shappe=(seq_length, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4) 