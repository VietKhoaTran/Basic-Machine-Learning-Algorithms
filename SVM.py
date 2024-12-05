import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn import svm

df= sklearn.datasets.load_iris()
y = df.target[0:100]
y = np.where(y == 0, 1, -1)
df = pd.DataFrame(df.data)
x = df.iloc[0:100, [0, 2]].values
k = svm.LinearSVC(C=1).fit(x, y)
def update_w_b (x, y, w, b, alpha):
    dl_dw = np.zeros(x.shape[1])
    dl_db = 0.0 
    for i in range(len(x)):
        zi = np.where(np.dot(x[i], w) +b >= 0, 1, -1)
        dl_dw += 2 * (zi - y[i]) * x[i] 
        dl_db += 2 * (zi - y[i]) 
    b -= dl_db * float(1/len(x)) * alpha
    w -= dl_dw * float(1/len(x)) * alpha   
    return w, b

def train(x, y, w, b, alpha, epoch):
    for i in range(epoch):
        w, b = update_w_b(x, y, w, b, alpha)
    return w, b
def draw(x_mew, w, b):
    a = -w[0] / w[1]
    return a*x_new -b/w[1]

w, b = train(x, y, np.array([0.0, 0.0]), 0.0, 0.1, 1000) 

x_new = []
for i in range (len(x)):
    x_new.append(x[i][0])
x_new = np.array(x_new)
plt.plot(x_new, draw(x_new, w, b), color = 'blue')
plt.scatter(x[:50, 0], x[:50, 1], color ='red', marker = 'o', label = 'Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color ='blue', marker = 'x', label = 'Veriscolor')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal width')
plt.legend(loc = 'upper left')
plt.show()