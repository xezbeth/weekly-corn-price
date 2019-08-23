import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,CuDNNLSTM,Dropout,Flatten
import numpy as np

class build_model:

    def __init__(self):

        self.model = Sequential()
        self.model.add(CuDNNLSTM(3,batch_input_shape=[None,None,None],return_sequence = False))
        self.model.compile(loss="mse",optimizer="adam",metrices=["accuracy"])

    def train(self,input,output,epochs = 10):

        return self.model.fit(input,output,epochs = epochs)
