import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,CuDNNLSTM,Dropout,Flatten,BatchNormalization
import numpy as np

class build_model:

    def __init__(self,input_shape):

        self.model = Sequential()
        self.model.add(CuDNNLSTM(2,input_shape=input_shape[1:],return_sequences=True))
        self.model.add(CuDNNLSTM(1,return_sequences=False))
        #self.model.add(CuDNNLSTM(1,return_sequences=False))
        #self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

        self.model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])


    def train(self,x_train,y_train,epochs = 10,validation=.1):

        return self.model.fit(x_train,y_train,epochs = epochs,validation_split = validation)

    def test(self,x_test,y_test):

        return self.model.evaluate(x_test,y_test)

    def predict(self,x_predict):

        return self.model.predict(x_predict)

    def save_model(self,path):

        self.model.save_weights(path)

    def load_model(self,path):

        return self.model.load_weights(path)

    def summary(self):

        return self.model.summary()
