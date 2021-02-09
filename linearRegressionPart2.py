import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

style.use("ggplot")

# the data is separated by semicolons so we specify
# the separation between the data using sep=";"
data = pd.read_csv("student-mat.csv", sep=";")

# define which attribute we're trying to predict
predict = "G3"

# we specify which columns we want out of the dataset
data = data[["G1", "G2", "absences", "failures", "studytime", "G3"]]
# to rearrange the data
data = shuffle(data)

# create 2 arrays using numpy x contains features, y contains labels
x = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # labels

# then split the data into testing and training data
# 90% of data is used to train
# 10% of data is used to test
# 0.1 specifies 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# this portion is to train the model multiple times and find the best score
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # this is to train and score the model using the arrays x and y created above
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print("Accuracy: " + str(accuracy))

    # in this case if the current model (the one we just tested
    # has a better model than the previous ones that have been tested
    # then we save it.
    if accuracy > best:
        # change the best to be the current accuracy
        best = accuracy
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

    # this is to load the model
    pickle_in = open("studentgrades.pickle", "rb")
    linear = pickle.load(pickle_in)

    # to check coefficient and intercept we can print it like this
    print('Coefficient: \n', linear.coef_)  # slope value
    print('Intercept:\n', linear.intercept_)  # this is the intercept

    # to check the predicted values we do this
    predictions = linear.predict(x_test)  # gets a list of all predictions
    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

    # then we plot the model
    # we can see different charts by changing which plot
    # we want to see, set plot to G1, G2, studytime, absences, and failures
    # in relation to G3 (which was predicted)
    plot = "absences"
    plt.scatter(data[plot], data["G3"])
    plt.legend(loc=4)
    plt.xlabel(plot)
    plt.ylabel("Final Grade")
    plt.show()