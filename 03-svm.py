# Lesson 8 - SVM - https://youtu.be/dNBvQ38MlT8

# Importing modules/packages
import sklearn
from sklearn import svm
from sklearn import datasets

cancer = datasets.load_breast_cancer()

print('Features:', cancer.feature_names)
print('Targets:', cancer.target_names)

# Splitting Data
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.2)

print(X_train[:5], y_train[:5])

# Lesson 10 - SVM - https://youtu.be/l2I8NycJMCY

# SVM
clf = svm.SVC(kernel='linear', C=2)
clf.fit(X_train, y_train)

# Metrics - Accuracy
from sklearn import metrics

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy - SVM', accuracy)

# KNN - Comparative
from sklearn.neighbors import KNeighborsClassifier

clf = clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(X_train, y_train)

# Metrics - Accuracy
from sklearn import metrics

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy - KNN', accuracy)

