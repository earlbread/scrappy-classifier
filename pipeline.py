from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from scrappy_classifier import ScrappyClassifier
my_classifier = ScrappyClassifier()

my_classifier.fit(X_train, y_train)

prediction = my_classifier.predict(X_test)
random_prediction = my_classifier.random_predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, prediction)
random_score = accuracy_score(y_test, random_prediction)
print('Scrappy Classifier Score = ', score)
print('Random Classifier Score = ', random_score)
