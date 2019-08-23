from model.model import build_model
import numpy as np
import math
import matplotlib.pyplot as plt
def sigmoid(num):
    #return 1/(1 + math.exp(-num))
    return num
def sigmoid1(num):
    return 1/(1 + math.exp(-num))

data = open('data/corn2013-2017.txt','r')
x_train = []
y_train = []
for d in data:
    x,y=d.split(',')
    x = x.split('-')
    xx = int(x[0]) * 365 + int(x[1]) * 30 + int(x[2])
    print(xx,"\n")
        #print(xxx)
    y = sigmoid(float(y.rstrip()))
    x_train.append(xx)

    #x_train.append(x)
    y_train.append(y)



x_train = np.array(x_train)
y_train = np.array(y_train)

x_train =x_train.reshape(248,1,-1)
#print("feat:" ,a,b,c , "lab:" , y , "\n")

model = build_model(input_shape=x_train.shape)
model.train(x_train,y_train,epochs=500)
x_predict = '2017-5-6'
x_predict = x_predict.split('-')
predict = []
for x in x_predict:
    predict.append(sigmoid(int(x)))
predict = np.array(predict)
predict =predict.reshape(1,3,1)
print(predict.shape)
res = model.predict(x_train)
plt.scatter(range(248),res,c='r')
plt.scatter(range(248),y_train,c='g')
plt.show()
