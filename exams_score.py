import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

exam  = pd.read_csv("https://raw.githubusercontent.com/NadavKiani/Students-Performance-in-Exams/master/StudentsPerformance.csv")

myencoder = LabelEncoder()

for column in exam.columns:
    if exam[column].dtype == "object":
        exam[column] = myencoder.fit_transform(exam[column])

print(exam.head())
# print(exam.isnull().any())
# print(exam.info())
print(exam["test preparation course"].unique())

exam.dropna(axis=0, inplace=True)

y = exam["test preparation course"].values
x = exam.drop("test preparation course", axis=1).values
