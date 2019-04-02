import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('diabetes.csv')

print(len(dataset))
print(dataset.head())

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
	dataset[column] = dataset[column].replace(0, np.NaN)
	mean = int(dataset[column].mean(skipna=True))
	dataset[column] = dataset[column].replace(np.NaN, mean)


# split dataset
x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.2)

# feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

import math
k = int(math.sqrt(len(y_test)))

if k % 2 == 0:
	k = k - 1

# Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=k, p=2,metric='euclidean')

# Fit the model
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)
print (y_pred)


# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)


print(f1_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))