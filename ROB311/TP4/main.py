#! /usr/bin/env python3
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


if __name__ == "__main__":
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = SVC()

    SVC.fit(X_train, y_train)

    score = SVC.score(X_test, y_test)
    print(f"Accuracy: {score}")
