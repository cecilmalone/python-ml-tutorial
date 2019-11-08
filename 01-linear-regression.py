# Lesson 2 - Linear Regression - https://youtu.be/45ryDIPHdGg

# Importing modules/packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, model_selection
from sklearn.utils import shuffle
# Lesson 4 - Save Models - https://youtu.be/3AQ_74xrch8
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Loading data
data = pd.read_csv('student-mat.csv', sep=';')

#print(data.head())

# Trimming data
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# Separating data
target = 'G3'

X = np.array(data.drop([target], axis=1)) # Features
y = np.array(data[target]) # Target

# Split and Test data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.1, random_state=42)

# Lesson 3 - Linear Regression - https://youtu.be/1BYu65vLKdA
# Lesson 4 - Save Models - https://youtu.be/3AQ_74xrch8

# Implementing Algorithm Linear Regression
# Train Model Multiple Time for Best Score

best = 0
for _ in range(20):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.1)

    linear = linear_model.LinearRegression()

    linear.fit(X_train, y_train)
    # Accuracy
    accuracy = linear.score(X_test, y_test)
    print('Accuracy:', str(accuracy))
    
    if accuracy > best:
        best = accuracy
        with open('studentgrades.pickle', 'wb') as f:
            pickle.dump(linear, f)

# Load Model
pickle_in = open('studentgrades.pickle', 'rb')
linear = pickle.load(pickle_in)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

# Predict
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

# Drawing and plotting model
plot = 'failures'
plt.scatter(data[plot], data[target])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel('Final Grade')
plt.show()
