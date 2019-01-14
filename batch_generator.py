
import tensorflow as tf
import numpy as np
from utils.gmm import mix_coef,gmm2d,gmm_loss
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import copy




def dataset_generator(inputs,seq_length):
    new_inputs = keras.preprocessing.sequence.pad_sequences(sequences=inputs, 
         maxlen=seq_length, 
         dtype='float32', 
         padding='post', 
         truncating='post', 
         value=0.0)
    return new_inputs
    
    
def batch_generator(inputs,batch_size,seq_length):    
    N = len(inputs)
    batch_train = [None]*batch_size
    batch_labels = [None]*batch_size
    for i in range(batch_size):
          index = np.random.randint(0, N)
          batch_train[i] = np.copy(inputs[index,0:seq_length])
          batch_labels[i] = np.copy(inputs[index,1:seq_length+1]) 
    return np.array(batch_train,dtype=np.float32), np.array(batch_labels,dtype=np.float32)


        