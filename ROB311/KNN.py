#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels


class KNN:
    # fonction qui permet d initialiser la classe
    # inputs : k, nombre de voisins pris en compte dans la methode
    def __init__(self, k):
        self.k = k

    # retourne le label predit pour une data
    # inputs : x, data
    def _predict(self, x):
        i_k = np.argpartition(self._distance(x), self.k)[:self.k]
        return np.argmax(np.bincount(self.y[i_k]))

    # calcule la distance d un point x avec tous les points du training set
    # inputs : x, data
    def _distance(self, x):
        return np.sqrt(np.sum((self.x-x)**2, axis=1))

    # fonction qui permet de definir les data et labels du training set
    # inputs : x, data
    #          y, labels
    def fit(self, x, y):
        self.x = x
        self.y = y

    # retourne le label predit pour x
    # inputs : x, data
    def predict(self, x):
        return np.array([self._predict(el) for el in x])

    # calcul le score du test set
    # inputs : x, data
    #          y, labels
    def score(self, x, y):
        y_pred = self.predict(x)
        return 100*sum(y_pred == y)/len(y)


# permet de charger les donnees suivant le data_set passer en argument de la
# ligne de commande
# inputs : path, le path vers le data_set
def load_breast_cancer_wisconsin():
    data = pd.read_csv("breast-cancer-wisconsin.data",
                       header=None,
                       na_values="?")
    data.dropna(inplace=True)
    for col in data.columns:
        data[col] = data[col].astype(int)

    x = data.values[:, 1:10]
    y = data.values[:, 10]

    return x, y

def load_haberman():
    data = pd.read_csv("haberman.data")
    x = data.values[:, 0:3]
    y = data.values[:, 3]

    return x, y

def plot_confusion_matrix(x, y, class_names):
    plt.title("Confusion matrix")
    ax = sns.heatmap(confusion_matrix(y_test, y_pred),
                     cmap=plt.cm.Blues,
                     annot=True,
                     fmt="d",
                     xticklabels=class_names,
                     yticklabels=class_names
    )
    ax.set(xlabel="True label", ylabel="Predicted label")

    return ax


if __name__ == "__main__":
    # Brest cancer Wisconsin dataset
    x, y = load_breast_cancer_wisconsin()
    class_names = [2, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    knn = KNN(6)
    knn.fit(x_train, y_train)
    acc = knn.score(x_test, y_test)
    y_pred = knn.predict(x_test)

    fig = plt.figure(f"Breast cancer Wisconsin dataset (accuracy={acc:.4}%)")
    plt.subplots_adjust(
        hspace=0.5,
        wspace=0.5
    )
    # Confusion matrix
    plt.subplot(2, 2, 1)
    plot_confusion_matrix(x, y, class_names)
    plt.subplot(2, 2, 2)
    plt.title("Feature 1 vs 5")
    ax = sns.scatterplot(x_test[:,1], x_test[:,5], hue=y_test)
    ax.set(xlabel="Feature 1", ylabel="Feature 5")
    plt.subplot(2, 2, 3)
    plt.title("Feature 1 vs 2")
    ax = sns.scatterplot(x_test[:,1], x_test[:,2], hue=y_test)
    ax.set(xlabel="Feature 1", ylabel="Feature 2")
    plt.subplot(2, 2, 4)
    plt.title("Feature 1 vs 5")
    ax = sns.scatterplot(x_test[:,1], x_test[:,5], hue=y_test==y_pred)
    ax.set(xlabel="Feature 1", ylabel="Feature 5")

    # Haberman dataset
    x, y = load_haberman()
    class_names = [1, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    knn = KNN(6)
    knn.fit(x_train, y_train)
    acc = knn.score(x_test, y_test)
    y_pred = knn.predict(x_test)

    plt.figure(f"Haberman dataset(accuracy={acc:.4}%)")
    plt.subplots_adjust(
        hspace=0.5,
        wspace=0.5
    )
    # Confusion matrix
    plt.subplot(2, 2, 1)
    plot_confusion_matrix(x, y, class_names)
    plt.subplot(2, 2, 2)
    plt.title("Feature 0 vs 1")
    ax = sns.scatterplot(x_test[:,0], x_test[:,1], hue=y_test)
    ax.set(xlabel="Feature 1", ylabel="Feature 5")
    plt.subplot(2, 2, 3)
    plt.title("Feature 1 vs 2")
    ax = sns.scatterplot(x_test[:,1], x_test[:,2], hue=y_test)
    ax.set(xlabel="Feature 1", ylabel="Feature 5")
    plt.subplot(2, 2, 4)
    plt.title("Feature 0 vs 2")
    ax = sns.scatterplot(x_test[:,0], x_test[:,2], hue=y_test)
    ax.set(xlabel="Feature 1", ylabel="Feature 5")

    plt.show()
