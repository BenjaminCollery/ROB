#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


if __name__ == "__main__":
    t0 = time.time()
    print("Loading the dataset...", end="")
    X_train = pd.read_feather("mnist_train.feather")
    y_train = X_train["label"]
    X_train.drop("label", axis="columns", inplace=True)
    X_test = pd.read_feather("mnist_test.feather")
    y_test = X_test["label"]
    X_test.drop("label", axis="columns", inplace=True)
    print("\tOk")

    print("Training the model...", end="")
    pipe = Pipeline([
        ("transformer", Normalizer()),
        ("classifier", SVC(gamma="scale"))])
    pipe.fit(X_train, y_train)
    print("\tOk")

    score = pipe.score(X_test, y_test)
    print(f"Accuracy: {score}")

    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    t1 = time.time()
    print(f"Finished in {t1-t0}s")

    sns.set()
    sns.heatmap(data=cm, annot=True)
    plt.title("Confusion matrix")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.show()
