#import bibilioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn . model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
from matplotlib.colors import ListedColormap
from tensorflow import keras
from sklearn.metrics import confusion_matrix

##################################################
#1. zadatak
##################################################

#učitavanje dataseta
data=pd.read_csv("2/titanic.csv")
data.drop_duplicates()
data=data.reset_index(drop=True)
print(data.info())
#a)
broj_zena=len(data[data['Sex']=='female'])
print("Broj zena iznosi:",broj_zena)
#b)
print("Postotak osoba koji nije prezivio potonuce:",(len(data[data['Survived']==0])/len(data))*100)
#c)
data_surviving_males=data[(data['Sex']=='male')&(data['Survived']==1)]
data_surviving_females=data[(data['Sex']=='female')&(data['Survived']==1)]
percent_of_surviving_males=(len(data_surviving_males)/len(data))*100
percent_of_surviving_females=(len(data_surviving_females)/len(data))*100
print(percent_of_surviving_males)
print(percent_of_surviving_females)
fig, ax = plt.subplots()
genders=["Male","Female"]
counts=[percent_of_surviving_males,percent_of_surviving_females]
plt.bar(genders,counts,color=['green','yellow'])
plt.xlabel("Spol")
plt.ylabel("Postotak prezivljavanja")
plt.title("Razlika postotka prezivljavanja muskaraca i zena")
plt.show()
#d)
print("Prosjecna starost prezivljelih zena:",data_surviving_females['Age'].mean())
print("Prosjecna starost prezivljelih muskaraca:",data_surviving_males['Age'].mean())
#e)
print(data[(data['Pclass']==1)&(data['Sex']=='male')&(data['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))
print(data[(data['Pclass']==2)&(data['Sex']=='male')&(data['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))
print(data[(data['Pclass']==3)&(data['Sex']=='male')&(data['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))
#???
##################################################
#2. zadatak
##################################################

#učitavanje dataseta
dataframe=pd.read_csv("2/titanic.csv")
dataframe.drop_duplicates()
dataframe=dataframe.reset_index(drop=True)

dataframe.info()
#train test split
ohe = OneHotEncoder()
input=['Sex','Embarked','Pclass','Fare']
X=dataframe[input]
y=dataframe['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sc = MinMaxScaler()
X_encoded_train = ohe.fit_transform(X_train[['Sex']]).toarray ()
X_encoded_test = ohe.fit_transform(X_test[['Sex']]).toarray ()



#a)
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_encoded_train,y_train)
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_encoded_train, y_train)
plot_decision_regions(X_encoded_train, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()  

#b)
y_test_p_KNN = KNN_model . predict ( X_encoded_test )
y_train_p_KNN = KNN_model . predict ( X_encoded_train )

print("KNN klasifikacija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))
#c)
scores = cross_val_score( KNN_model , X_encoded_train , y_train , cv =5 )
print(scores)
#d)
KNN_model = KNeighborsClassifier(n_neighbors=2)
KNN_model.fit(X_encoded_train,y_train)
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_encoded_train, y_train)
plot_decision_regions(X_encoded_train, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()  
y_test_p_KNN = KNN_model . predict ( X_encoded_test )
y_train_p_KNN = KNN_model . predict ( X_encoded_train )

print("KNN klasifikacija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

##################################################
#3. zadatak
##################################################

#učitavanje podataka:
dataframe=pd.read_csv("titanic.csv")
dataframe.drop_duplicates()
dataframe=dataframe.reset_index(drop=True)
#train test split
ohe = OneHotEncoder()
input=['Sex','Embarked','Pclass','Fare']
X=dataframe[input]
y=dataframe['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc = MinMaxScaler()
X_encoded_train = ohe.fit_transform(X_train[['Sex']]).toarray ()
X_encoded_test = ohe.fit_transform(X_test[['Sex']]).toarray ()
num_classes = 10
(X_encoded_train, y_train), (X_encoded_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = X_encoded_train.astype("float32") / 255
x_test_s = X_encoded_test.astype("float32") / 255
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

#a)
model=keras.Sequential()
model.add(layers.Input(shape=(891,)))
model.add(layers.Dense(12,activation='relu'))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

#b)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#c)
batch_size = 5
epochs = 100
model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#d)
model.save("ispit_model_keras.keras")
#e)
score=model.evaluate(x_train_s,y_test_s,verbose=0)
#f)
predictions=model.predict(x_test_s)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
cm = confusion_matrix(y_test, predictions.argmax(axis=1))
print(cm)
