#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

# permet de charger les donnees suivant le data_set passer en argument de la ligne de commande
# inputs : path, le path vers le data_set
def load_data(path):
    data = pd.read_csv(path,
                       header=None,
                       na_values="?")
    data.dropna(inplace=True)
    for col in data.columns:
        data[col] = data[col].astype(int)

    if "haberman" in path:
        x = data.values[:, 0:3]
        y = data.values[:, 3]
        class_names = [1,2]
    else:
        x = data.values[:, 1:10]
        y = data.values[:, 10]
        class_names = [2,4]

    return x, y, class_names

# code venant d un exemple fourni par sklearn
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

# affiche la matrice de confusion et ......
def plots(y_true, y_pred):
    plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

####################
if __name__ == "__main__":
    data_set_path = sys.argv[1]
    x, y, class_names = load_data(data_set_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    k = 6

    knn = KNN(k)
    knn.fit(x_train, y_train)
    acc = knn.score(x_test, y_test)

    y_pred = knn.predict(x_test)

    print(f"Accuracy:  {acc:.4}%")

    plots(y_test, y_pred)

# david.octavian.iacob@gmail.com
