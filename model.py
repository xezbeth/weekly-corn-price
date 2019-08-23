import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,CuDNNLSTM,Dropout,Flatten
import numpy as np

class build_model:

    def __init__(self):

        self.model = Sequential()
        self.model.add(CuDNNLSTM(3,batch_input_shape=[None,None,None],return_sequence = False))
        self.model.compile(loss="mse",optimizer="adam",metrices=["accuracy"])

    def train(self,x_train,y_train,epochs = 10):

        return self.model.fit(x_train,y_train,epochs = epochs)

    def test(self,x_test,y_test):

        return self.model.evaluate(x_test,y_test)

    def predict(self,x_predict):

        return self.model.predict(x_predict)

    def save_model(self,path):

        self.model.save_weights(path)

    def load_model(self,path):

        return self.model.load_weights(path)
