import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def dist(self):
        return np.sqrt((self.x-self.y)**2)

    def predict(self, y):
        dist = dist()
        argdist = np.argpartition(dist , k)[:k]
        return np.bincount(y[argdist])

####################
if __name__ == "__main__":
    data = pd.read_csv("breast-cancer-wisconsin.data")
    print(data)
    x = data.values[1:10]
    y = data.values[10]
    x_train , y_train , x_test , y_test = train_test_split(x , y)

    k = 10

    knn = knn(k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(y_test[0])

    print("y_test :")
    print(y_test[0])
    print("y_pred :")
    print(y_pred)

### david.octavian.iacob@gmail.com
