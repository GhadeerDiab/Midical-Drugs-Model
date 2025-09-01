#decision tree and over fitting
import  pandas as pd
data = pd.read_csv("drug200.csv")

data["Sex"] = data["Sex"].map({"M":0, "F":1})
data["BP"] =  data["BP"].map({"LOW":0, "NORMAL":1,"HIGH":2})
data["Cholesterol"] = data["Cholesterol"].map({"NORMAL":0, "HIGH":1})

x = data[["Age", "Sex", "BP", "Cholesterol","Na_to_K"]]
y = data["Drug"]

import sklearn.tree as tr

drgTree = tr.DecisionTreeClassifier()
drgTree.fit(x,y)

import matplotlib.pyplot as plt
cols = [ "Age", "Sex", "Bp", "Cholesterol", "Na_to_K"]
cls = drgTree.classes_
tr.plot_tree(drgTree,class_names=cls,feature_names=cols)

plt.show()
print(tr.export_text(drgTree,class_names=cls,feature_names=cols))

from sklearn.model_selection import train_test_split

xtr, xtest, ytr, ytest = train_test_split(x,  y , test_size=0.3)



drgTree.fit(xtr, ytr)

ytr_pred  = drgTree.predict(xtr)
ytest_pred = drgTree.predict(xtest)

import sklearn.metrics as mtr
print(mtr.accuracy_score(ytr,ytr_pred))
print(mtr.accuracy_score(ytest,ytest_pred))


















