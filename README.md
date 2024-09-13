# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Print the present data and placement data and salary data.
3.Using logistic regression find the predicted values of accuracy confusion matrices.
4.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JABEZ S
RegisterNumber: 212223040070

import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
*/
```

### Output:
## PLACEMENT DATA:
![image](https://github.com/user-attachments/assets/f5a42273-db65-4dbb-9d07-15fb772ea451)

## Checking the null() function:
![Screenshot 2024-09-13 143343](https://github.com/user-attachments/assets/44cb2aca-1e95-450d-9d88-c534a275f2aa)

## Print Data:
![image](https://github.com/user-attachments/assets/826479d7-d4a0-4cf6-97a1-0233edf735f6)


## Y Prediction:
![image](https://github.com/user-attachments/assets/70c7a5a2-fc58-4e42-9093-d973addff1b0)

## Accuracy Value:
![Screenshot 2024-09-13 143944](https://github.com/user-attachments/assets/7fe9fa42-ef82-433e-bec0-88fd20294e08)
 
 ## Confusion Array:
 ![Image-6](https://github.com/user-attachments/assets/94082e09-763d-4575-ae59-10dfffeffcb1)

## Classification Report:
![Image-7](https://github.com/user-attachments/assets/25ec8f64-2d4c-47b8-97aa-1cde704fb99c)

 ## Prediction Of LR:
  ![Image-8](https://github.com/user-attachments/assets/6bbf7b16-9da2-4b8e-b404-8d51d56fba09)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
