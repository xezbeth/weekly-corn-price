
data = open('data/corn2013-2017.txt','r')
x_train = []
y_train = []
for d in data:
    x,y=d.split(',')
    x = x.split('-')
    x_train.append(x)
    y_train.append(y)