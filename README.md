# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
    1.Import the required packages.
    2.Read the data set.
    3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
    4.Determine training and test data set.
    5.Apply decision tree Classifier and get the values of accuracy and data prediction.
```

## Program:
```
/*
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![image](https://user-images.githubusercontent.com/108709865/203898366-7eaf2ff1-e212-4b02-aeaa-33ae0b2c8dd3.png)

![image](https://user-images.githubusercontent.com/108709865/203898477-ff6ca350-1278-4cce-9e01-1e57afa8e4c7.png)

![image](https://user-images.githubusercontent.com/108709865/203898576-e7911cba-59d2-43c2-9db9-c674f822f1f7.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
