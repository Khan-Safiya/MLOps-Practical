import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report, confusion_matrix

df=pd.read_csv("Admission_Predict.csv")
print("Dataset Preview:\n", df.head())

df.rename(columns={"Chance of Admit ":"Chance"},inplace=True)
df['Admitted']=np.where(df['Chance']>=0.5,1,0)

X=df[["GRE Score","CGPA"]]
y=df["Admitted"]
model=DecisionTreeClassifier()
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=64)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_pred,y_test))
print("Precision Score:", round(precision_score(y_pred,y_test),3))
print("Recall Score:", round(recall_score(y_pred,y_test),3))
print("\nConfusion Matrix:\n", confusion_matrix(y_pred,y_test))
print("\nClassification Report:\n", classification_report(y_pred,y_test))

print("Actual Values: \n", y_test[:10].values)
print("Predicted Values: \n", y_pred[:10])

gre=float(input("\nEnter the GRE Score (out of 340): "))
cgpa=float(input("Enter the CGPA (out of 10): "))
y_pred=model.predict([[gre,cgpa]])
if y_pred[0]==1:
    print("\nThe student is likely to get admission.")
else:
    print("\nThe student is unlikely to get admission.")
