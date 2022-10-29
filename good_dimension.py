import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=3000, n_features=30, n_informative=4, n_redundant=26, n_classes=3, random_state=7);
# Lecture hour 1
# data = pd.DataFrame(X, columns=["Feature " + str(k) for k in range(30)])
#
# data["target"] = y
#
# print(data.head())
#
# print(data.corr())

# Lecture hour 2

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3);

classifier = KNeighborsClassifier()

classifier.fit(x_train, y_train)

print(classifier.score(x_train, y_train))

# pca = PCA(0.9);

pca = PCA(n_components=3)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x_train.shape)
print(np.sum(pca.explained_variance_ratio_))


