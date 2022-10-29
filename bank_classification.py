import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


dataUrl = "https://raw.githubusercontent.com/Shwetabhdixit/Knn-for-bank-classification-dataset/master/bank.csv"

bank = pd.read_csv(dataUrl, sep=";")
myLabel = LabelEncoder()

for column in bank.columns:
    if bank[column].dtype == "object":
        bank[column] = myLabel.fit_transform(bank[column])

y = bank["y"].values
x = bank.drop("y", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

classifier = KNeighborsClassifier(n_neighbors=7, weights="distance")
classifier.fit(X_train, y_train)

print(classifier.score(X_train, y_train))
print(classifier.score(X_test, y_test))
print(bank.loc[0, :].values)

new_data = np.array([bank.loc[0, :].values[:-1]])

# print(classifier.predict(new_data))


