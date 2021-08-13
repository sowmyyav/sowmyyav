# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:25:00 2021

@author: Sowmya
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold
import tensorflow as tf
rsp_deap_data, rsp_deap_label = joblib.load(open('C:/Users/Sowmya/Desktop/LSTM/lstm_slider128_rsp_16fs_64overlap_nobaseline.dat', 'rb'))

#convert raw label into categorical data- this generates 10 classes
from tensorflow.keras.utils import to_categorical

def data_binarizer(ratings, threshold1, threshold2):
	"""binarizes the data below and above the threshold"""
	binarized = []
	for rating in ratings:
		if rating < threshold1:
			binarized.append(0)
		elif rating>= threshold2:
			binarized.append(1)
	return binarized

#convert binarized label (0 and 1) into categorical data- this generates 2 classes
y_valence = np.array(data_binarizer([el[0] for el in rsp_deap_label],5,5))
Z1 = np.ravel(y_valence)
y_train1 = to_categorical(Z1)
y_train1
#639ms/step - loss: 0.6824 - acc: 0.5666 - val_loss: 0.6833 - val_acc: 0.5676

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_valence)
print(counter)

#use stratify for split   
X_train_rsp_val, X_test_rsp_val, y_train_rsp_val, y_test_rsp_val = train_test_split(rsp_deap_data, y_train1, test_size=0.2, random_state=42, stratify=y_train1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(X_train_rsp_val)
testing_set_scaled = sc.transform(X_test_rsp_val)

#sc.data_min_
#sc.data_max_

x_train = training_set_scaled.reshape(training_set_scaled.shape[0],training_set_scaled.shape[1], 1)
x_test = testing_set_scaled.reshape(testing_set_scaled.shape[0],testing_set_scaled.shape[1], 1)

from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Permute
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking
#from keras.utils import plot_model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Activation
from sklearn.model_selection import train_test_split


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import optimizers 


inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
#shape = Permute((2, 1))(inputs)
    
input_shape = (x_train.shape[1], 1)

model = Sequential()
 
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(units = 256, return_sequences = True))  
model.add(Dropout(0.3))     
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))  
model.add(Dense(2))
model.add(Activation('softmax'))
    
model.compile(optimizer ="adam", loss =keras.losses.categorical_crossentropy,metrics=["acc"])
model.summary()
model.fit(x_train, y_train_rsp_val,epochs=10,batch_size=256,verbose=1,validation_data=(x_test, y_test_rsp_val))