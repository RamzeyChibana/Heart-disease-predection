from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os


dataset=pd.read_csv('heart.csv')

os.system('cls')

print(dataset)

#-----------------Data preparation---------------------------------------------------------------

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

kf=KFold(n_splits=4,shuffle=True,random_state=1)
print("x shape:",X.shape,"| y shape:",y.shape)

pred=np.zeros(y.shape[0])

acc=[]

for tr,ts in kf.split(X,y):
    X_train=X[tr]
    X_test=X[ts]
    y_train=y[tr]
    y_test=y[ts]
    #training
    clModel=LogisticRegression(solver="liblinear")
    clModel.fit(X_train,y_train)
    #testing
    y_pred=clModel.predict(X_test)
    pred[ts]=y_pred
    acc.append(accuracy_score(y_test,y_pred))
    print("accuracy:",accuracy_score(y_test,y_pred))

#metrices
print('total pred:',accuracy_score(y,pred))
print("mean of accuracy",np.mean(acc))
print("f1 score",f1_score(y,pred))
print("confusion matrix \n",confusion_matrix(y,pred))