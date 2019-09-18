import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, k):
        self.k = k

    def _predict(self, x):
        i_k = np.argpartition(self._distance(x), self.k)[:self.k]
        return np.argmax(np.bincount(y[i_k]))

    def _distance(self, x):
        return np.sqrt(np.sum((self.x-x)**2, axis=1))

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        return np.array([self._predict(el) for el in x])

    def score(self, x, y):
        y_pred = self.predict(x)
        return 100*sum(y_pred == y)/len(y)


def load_data():
    data = pd.read_csv("breast-cancer-wisconsin.data",
                       header=None,
                       na_values="?")
    data.dropna(inplace=True)
    for col in data.columns:
        data[col] = data[col].astype(int)
    x = data.values[:, 1:10]
    y = data.values[:, 10]

    return x, y


####################
if __name__ == "__main__":
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    k = 10

    knn = KNN(k)
    knn.fit(x_train, y_train)
    acc = knn.score(x_test, y_test)
    print(f"Accuracy:  {acc:.4}%")

# david.octavian.iacob@gmail.com
