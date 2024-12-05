import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = sklearn.datasets.load_iris()
y = df.target[0:100]
y = np.where(y == 0, 1, -1)
df = pd.DataFrame(df.data)
x = df.iloc[0:100, [0, 2]].values


clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(x, y)

new_data = [[5.0, 2.4]]
print(clf.predict(new_data))
plot_tree(clf, filled = True, feature_names = ['Length','Width'], class_names = ['-1', '1'], rounded = True)
plt.title('Decision Tree')
plt.show()