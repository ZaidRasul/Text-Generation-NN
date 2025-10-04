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

