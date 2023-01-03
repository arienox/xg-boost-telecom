import os
import pandas as pd
os.chdir('C:/Users/ASUS/OneDrive/Documents/Python ML')
teledata = pd.read_csv('Telecom_data.csv')
print(teledata)
print(teledata.head(10))
print(teledata.isnull().sum())
print(teledata.dtypes)

x = teledata.iloc[:,:-1]
y = teledata.iloc[:,-1:]

x= pd.get_dummies(x)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
model = XGBClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)   # X_test is input variable 
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
    
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test, y_test) 

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

from sklearn.metrics import accuracy_score
a1 = accuracy_score(y_test, y_pred)
print("Accuracy score : {:.2f}%".format(a1*100))


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
(tn,fp,fn,tp)
