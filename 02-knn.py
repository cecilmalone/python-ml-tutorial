# Lesson 5 - KNN - https://youtu.be/ddqQUz9mZaM

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

# Load data
data =  pd.read_csv('car.data')

# Converting data
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clss = le.fit_transform(list(data['class']))

# Recombine data
X = list(zip(buying, maint, door, persons, lug_boot, safety)) #features
y = list(clss) #target

# Train e test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.1)

# Lesson 6 - KNN - https://youtu.be/TQKI0KE-JYY

# Implementation KNN
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(accuracy)

# Testing Model
predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], 'Data: ', X_test[x], 'Actual: ', names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9, True)
    print("N: ", n)
