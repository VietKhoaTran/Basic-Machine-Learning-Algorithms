import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

df = sklearn.datasets.load_iris()
df = pd.DataFrame(df.data)
x = df.iloc[0:50, 0].values
y = df.iloc[0:50, 1].values

print(x)

def update_w_and_b(x, y, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(x)
  
    for i in range(N):
        dl_dw += -2 * x[i] * (y[i] - (w*x[i] + b))      
        dl_db += -2 * (y[i] - (w*x[i] + b))
    w = w - (1/float(N)) * dl_dw * alpha
    b = b - (1/float(N)) * dl_db * alpha
    return w, b 

def train(x, y, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_and_b(x, y, w, b, alpha)
        if e % 400 == 0:
            print("epoch: ", e, "loss: ", avg_loss(x, y, w, b))
    return w, b

def avg_loss(x, y, w, b):
    N = len(x)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (w * x[i] +b)) ** 2
    return total_error/ float(N)

def predict(x, w, b):
    return w*x +b

w, b = train(x, y, 0.0, 0.0, 0.001, 1500)
x_new = 4.6
print(predict(x_new, w, b))
print(x)
plt.scatter(x, y, color = 'blue', marker = 'o', label = 'Sepal Length/ Sepal Width')
plt.plot(x, predict(x, w, b))
plt.xlabel('Sepal Lenth')
plt.ylabel('Sepal Width')
plt.legend(loc = 'upper left')
plt.show()
