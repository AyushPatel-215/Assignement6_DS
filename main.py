import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def readFile_and_fetchData():
    global data, X, y
    data = pd.read_csv('Assignment6_data.csv')
    data.info()
    print("Duplicate rows:", len(data[data.duplicated()]))

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = pd.get_dummies(X, columns=['State'], drop_first=True)


def train_model():
    global y_test, y_pred, model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


def find_performance():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)


def draw_plot():
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')


# def predict_value():
#     predicated_value = model.predict([[165349.20, 136897.80, 471784.10, 0, 1]])
#     print("predicted value is: ", predicated_value[0])


def make_pickle_file():
    with open('model.pickle', 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    readFile_and_fetchData()
    train_model()
    find_performance()
    draw_plot()
    # predict_value()
    make_pickle_file()
