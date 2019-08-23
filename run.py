from model.model import build_model
import numpy as np

data = open('data/corn2013-2017.txt','r')
x_train = []
y_train = []
for d in data:
    x,y=d.split(',')
    x = x.split('-')
    x_train.append(x)
    y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
x_train =x_train.reshape(248,3,-1)


model = build_model(input_shape=x_train.shape)
model.train(x_train,y_train,epochs=500)
