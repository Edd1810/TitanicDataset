import pandas as pd
import pydotplus as pp
from sklearn.tree import DecisionTreeClassifier, export_graphviz

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

#Spit the tree out
lab = train["Survived"]
feat = train.drop("Survived", axis='columns')
tree = DecisionTreeClassifier(min_samples_split=50, random_state=5)
tree.fit(feat, lab)
print(tree)

DTdata = export_graphviz(
    tree,
    feature_names=list(feat),
    class_names="Survived",
    out_file=None,
    rounded=True,
    filled=True
    )

DT = pp.graph_from_dot_data(DTdata)
DT.write_pdf("Tree.pdf")

# Prediction
pred = tree.predict(test)
print(pred)

output = pd.DataFrame({'Survived': pred})
output.to_csv("Survival_Prediction.csv")
