#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    digits = pd.read_csv("optdigits.tra", header=None).values
    X = digits[:,:-1]
    y = digits[:,-1]

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)

    for i in range(10):
        pred = digits[kmeans.labels_==i][:,-1]
        print(f"Predicted class {i}")
        print(f"\tmost common element: {np.argmax(np.bincount(pred))} ({np.max(np.bincount(pred))} of {len(pred)})")

    sns.heatmap(confusion_matrix(y, kmeans.labels_), annot=True)
    plt.title("Confusion matrix")
    plt.xlabel("True class")
    plt.ylabel("Predicted class")
    plt.show()
