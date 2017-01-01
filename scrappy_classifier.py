import random

from scipy.spatial import distance


def euc(a, b):
    """Return distance between a and b
    """
    return distance.euclidean(a, b)

class ScrappyClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def random_predict(self, X_test):
        predictions = []

        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)

        return predictions
