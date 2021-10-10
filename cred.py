import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dataset=pd.read_csv('creditcard.csv')
#print(dataset)




dataset[dataset.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6)
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
dataset[dataset.Class == 1].Time.hist(bins=35, color='red', alpha=0.6, label="Fraudulant Transaction")
#plt.legend()
plt.subplot(2, 2, 2)
dataset[dataset.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6, label="Non Fraudulant Transaction")
#plt.legend()
#plt.show()

#fraud=dataset.loc[dataset['Class']==1]
#print('Total number of fraud transactions:',len(fraud))
#legit=dataset.loc[dataset['Class']==0]
#print('Total number of legit transactions:',len(legit))

#dataset.info()

x=dataset.iloc[:,:-1].values
y=dataset['Class']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)
data=pd.DataFrame(x_train,y_train)
from sklearn.linear_model import LogisticRegression
lreg=LogisticRegression()
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)

#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(pd.DataFrame(y_pred))

ypred1=lreg.predict([[6000,1.49998742,1.221636809,-1.3830151,1.234898688,0.5419474,-0.753230165,-0.6404975,-0.227487228  ,
                      1.04010573, 1.323729274 ,3.227666231,-0.242681999,1.205416808,6,0.1,-2.815612186,0.873936448,
                      -0.847788599,-0.683192626,-0.102755942,2.180239,1.48328533 ,0.011 ,0.392830885 ,0.156134554 ,-1.35499004 ,
                      0.026415549 ,0.042422089,250]])

print(ypred1)



from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

#cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
#print(cnf_matrix)
#print(classification_report(y_test,y_pred))
#sns.heatmap(pd.DataFrame(cnf_matrix),annot=True ,fmt='g')
#plt.title("confusion matrix")
#plt.xlabel("actual label")
#plt.ylabel("predicted label")
#plt.show()


