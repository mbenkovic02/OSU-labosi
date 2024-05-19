import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Pod a
plt.scatter(X_train[:,0], X_train[:,1], c="red")
plt.scatter(X_test[:,0], X_test[:,1], c="blue", marker='*')
'''
plt.scatter(X_train[:,0], y_train, c="red")
plt.scatter(X_train[:,1], y_train, c="red")
plt.scatter(X_test[:,0], y_test, c="blue", marker='*')
plt.scatter(X_test[:,1], y_test, c="blue", marker='*')
'''
plt.show()

# Pod b
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
y_test_p = LogRegression_model . predict ( X_test )

# Pod c
b = LogRegression_model.intercept_[0]
w1,w2 = LogRegression_model.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])
yd = m*xd+c

plt.plot(xd,yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin,color='orange', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='blue', alpha=0.2)
plt.show()

# Pod d
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()
print(classification_report(y_test , y_test_p))

#Pod e
y1 = (y_test==y_test_p)
y0 = (y_test!=y_test_p)

X_false = []

for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        X_false.append([X_test[i, 0], X_test[i, 1]])

X_false = np.array(X_false)
print(X_false)

plt.scatter(X_test[:,0], X_test[:, 1])
plt.scatter(X_false[:,0], X_false[:,1], color='green')
plt.show()
