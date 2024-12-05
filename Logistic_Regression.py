import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import math

df = sklearn.datasets.load_iris()
y = df.target[50:150]
y = np.where(y==1, 0, 1)
print(y)
df = pd.DataFrame(df.data)
x = df.iloc[0:100, 3].values
errors = []
def update_w_b (x, y, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    for i in range(len(x)):
        value = y[i] * (1 / (1 + np.exp(w * x[i] +b))) + (1 - y[i]) * (-1 / (1 + np.exp(w * x[i] +b))) 
        dl_dw += x[i] * value
        dl_db += value
    b += dl_db * float(1/len(x)) * alpha
    w += dl_dw * float(1/len(x)) * alpha
    return w, b
def train(x, y, w, b, alpha, epoch):
    for i in range(epoch):
        w, b = update_w_b(x, y, w, b, alpha)
    return w, b

def draw(x, w, b):
    return 1 / (1 + np.exp(-w*x-b))

w, b = train(x, y, 0.0, 0.0, 0.1, 2000)
print(w, b)
x_range = np.linspace(min(x), max(x), 300)
plt.plot(x_range, draw(x_range, w, b), color = 'green')
plt.scatter(x[:50], y[:50], color = 'red', marker = 'x', label ='Veriscolor')
plt.scatter(x[50:100], y[50:100], color = 'blue', marker = 'o', label ='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Type of flower')
plt.legend(loc = 'upper left')
plt.show()