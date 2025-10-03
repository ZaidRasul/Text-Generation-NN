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