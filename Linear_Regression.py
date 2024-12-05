import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

df = sklearn.datasets.load_iris()
df = pd.DataFrame(df.data)
print(df)
x = df.iloc[0:50, 0].values
y = df.iloc[0:50, 1].values
print(x)
def update_w_b (x, y, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    for i in range (len(x)):
        dl_dw += -2 * x[i] * (y[i] - w * x[i] -b)
        dl_db += -2 * (y[i] - w * x[i] -b)
    w -= dl_dw * float(1 / len(x)) * alpha
    b -= dl_db * float(1 / len(x)) * alpha 
    return w, b

def train(x, y, w, b, alpha, epoch):
    for i in range(epoch):
        w, b = update_w_b(x, y, w, b, alpha)
    return w, b

def predict(x, w, b):
    print(w)
    return w*x + b

w, b = train(x, y, 0.0, 0.0, 0.001, 20000)
plt.plot(x, predict(x, w, b), color = 'blue')
print(x)
def train_2(x, y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x, y)
    return model
x = x.reshape(-1, 1)
model = train_2(x, y)

plt.plot(x, model.predict(x), color = 'green', ls = '--')
plt.scatter(x, y, color = 'red', marker = 'o', label ='length/width')
plt.xlabel('Petal width')
plt.ylabel('Petal length')
plt.legend(loc = 'upper left')
plt.show()