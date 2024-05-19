import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# stvarna vrijednost izlazne velicine i predikcija
y_true = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 1, 0, 0])
# tocnost
print("Tocnost: " , accuracy_score(y_true ,y_pred))
# matrica zabune
cm = confusion_matrix(y_true, y_pred)
print ("Matrica zabune: " , cm)
disp = ConfusionMatrixDisplay( confusion_matrix(y_true , y_pred))
disp.plot()
plt.show()
# report
print(classification_report(y_true , y_pred))