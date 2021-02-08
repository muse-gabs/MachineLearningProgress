import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# the data is separated by semicolons so we specify
# the separation between the data using sep=";"
data = pd.read_csv("student-mat.csv", sep=";")

# this printed the entire dataset (cut off some of the data in the middle
#print(data.head())

# we specify which columns we want out of the dataset
data = data[["G1","G2", "G3", "studytime", "failures", "absences"]]

# then print the result
print(data.head())

# define which attribute we're trying to predict
predict = "G3"

# create 2 arrays using numpy x contains features, y contains labels
x = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # labels

# then split the data into testing and training data
# 90% of data is used to train
# 10% of data is used to test
# 0.1 specifies 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# this is to define the model
linear = linear_model.LinearRegression()

# this is to train and score the model using the arrays x and y created above
linear.fit(x_train,y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy)

# to check coefficient and intercept we can print it like this
print('Coefficient: \n', linear.coef_) # slope value
print('Intercept:\n', linear.intercept_) # this is the intercept

# to check the predicted values we do this
predictions = linear.predict(x_test) # gets a list of all predictions

print('These are the predictions: \n')
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])