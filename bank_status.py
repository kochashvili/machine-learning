import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  LabelEncoder

bank = pd.read_csv("https://raw.githubusercontent.com/SheepShaun/Bank-customer-churn-prediction/main/bank%20customer%20churn%20dataset.csv")

encoder = LabelEncoder()
bank.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

for column in bank.columns:
    if bank[column].dtype == "object":
        bank[column] = encoder.fit_transform(bank[column])

print(bank.head())

y = bank['IsActiveMember'].values
X = bank.drop("IsActiveMember", axis=1).values

X2 = bank[["CreditScore", "Geography", "Gender", "Exited"]].values


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)

algo1 = LogisticRegression(max_iter=2000000)
algo2 = GaussianNB(priors=[0.3, 0.7])
algo1.fit(X_train, y_train)
algo2.fit(X_train, y_train)

print(algo1.score(X_test))
print(algo2.score(y_test))





print(bank.head())