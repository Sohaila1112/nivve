import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
////////
df = pd.read_csv('/content/Breast Cancer data.csv')
df
////
X = df.drop(['id', 'diagnosis'] ,   axis = 1)
X
////
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
X
////
y = np.array(df['diagnosis'])
y
///

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2)
print(len(X_train))
print(len(X_test))
//
clf = GaussianNB()
clf.fit(X_train, y_train)
//
y_pred = clf.predict(X_test)
//
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


///
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
recall = recall_score(y_test, y_pred)
print('Recall:', recall)
precision = precision_score(y_test, y_pred)
print('Precision:', precision)
f1 = f1_score(y_test,y_pred)
print('F1_score:', f1)
///
cm = confusion_matrix(y_test, y_pred)
cm# nivve
