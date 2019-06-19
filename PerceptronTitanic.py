from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np


train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("titanic_test.csv")

# --- Cleaning ---
train = train.drop(["PassengerId", "Name", "Parch", "Ticket", "Fare","Cabin","Embarked"], axis=1)
# FILL NA DONT DROP
train = train.fillna(0)
# Turn genders into 1's or 0's
gender = {'female': 0, 'male': 1}
train = train.replace({'female': gender, 'male': gender})
train.Sex =[gender[item] for item in train.Sex]
# --- End of Cleaning ---

# --- TEST Cleaning ---
test = test.drop(["PassengerId", "Name", "Parch", "Ticket", "Fare","Cabin", "Embarked"], axis=1)
test = test.fillna(0)
# Turn genders into 1's or 0's
gender = {'female': 0, 'male': 1}
test = test.replace({'female': gender, 'male': gender})
test.Sex =[gender[item] for item in test.Sex]
# --- End of TEST Cleaning ---

label = train.Survived
features = train.drop("Survived", axis='columns')

p = Perceptron()
p.fit(features, label)
print(p.fit(features, label))
p.predict(test)
print(p.predict(test))