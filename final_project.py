# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/creditcard.csv') #read csv data
print("Original shape:", df.shape) #check shape

#find missing values in data set
df.info()

print(df.head()) #first few rows print from the Dataset

# Convert 'Time' column to integer
df['Time']=df['Time'].astype(int)

df.info()

df.hist(figsize=(20,20))
plt.show()

"""Exploratory data analysis EDA"""

C=df['Class'].value_counts()
print(C)

from importlib import reload
plt=reload(plt)
C=df['Class'].value_counts()
print('Normal CC',C[0])
print('Fraud CC',C[1])
plt.figure(figsize=(10,5))
sns.countplot(x='Class',data=df,color='blue')
plt.title('Class result')
plt.legend(['Normal','Fraud'])
plt.xlabel("Normal result", color='red')
plt.ylabel('Fraud result',color='red')
plt.xticks([0,1],['Normal','Fraud CC'],color='green')
plt.show()

df.columns

#Feature selection
c=df.corrwith(df['Class'])

df.drop('Class',axis=1).corrwith(df['Class']).plot(kind='bar',figsize=(12,6))

print(c.sort_values(ascending=False))

f=df.corr()

plt.figure(figsize=(20,10))
t=c.sort_values(ascending=False)
sns.barplot(x=t.values,y=t.index)
plt.title('correlation with target')
plt.show()

"""Analyze Target Variable Distribution"""

target_column='Class'
class_counts=df[target_column].value_counts()
total_data=len(df)
class_percentages=(class_counts/total_data)*100
print('counts',class_counts)
print('percentages',class_percentages)

#check data is imbalanced
t=0.4
min_percentages=class_percentages.min()
max_percentages=class_percentages.max()
if min_percentages/max_percentages<t:
  print('data is imbalanced')
else:
  print('data is balanced')

"""#find and handling outlires"""

#outlier
h=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class']
x=df[h]

#outlires
#using IQR method --- to detect the outlires
def outlires(df,column):
  q1=df[column].quantile(0.25)
  q3=df[column].quantile(0.75)
  iqr=q3-q1
  lower=q1-1.5*iqr
  upper=q3+1.5*iqr
  outlires=df[((df[column]<lower)|(df[column]>upper))]
  return outlires

#outlires columnwise
for col in h:
  outlires1 = outlires(df,col)
  print(f"column:{col},'number of outlires\n='{len(outlires1)}")

df.describe()

plt.figure(figsize=(20,5))
plt.title('Detection of outlires')
sns.boxplot(data=df[h])
plt.show()

#handle outlires(cliping)
def clip(df,column):
  q1=df[column].quantile(0.25)
  q3=df[column].quantile(0.75)
  iqr=q3-q1
  lower=q1-1.5*iqr
  upper=q3+1.5*iqr
  df[column]=df[column].clip(lower=lower,upper=upper)
  return df

# Apply IQR-based capping only on features, not on 'Class'
for col in df.columns:
    if col != 'Class':
        df = clip(df, col)

#y={}
#for column in h:
#  df=clip(df,column)
#  outlires1 = outlires(df,column)
#  y[column]=outlires1
#  print(f"column:{col},'number of outlires\n=',{len(df)}")

print(f"column:{col},'number of outlires'\n= {len(outlires1)}")
print(df['Class'].value_counts())

plt.figure(figsize=(20,5))
plt.title('After capping')
df[h].boxplot()
plt.show()

#find duplicate records ?
dupl_count = df.duplicated().sum()
print("The number of duplicates:", dupl_count)

# Drop duplicate rows based on feature columns only (preserve 'Class')
df = df.drop_duplicates(subset=df.columns.difference(['Class']))
print("After removing duplicates:", df.shape)

"""#banchmark dataset"""

df.to_csv('finaldataset.csv',index=False)

df1=pd.read_csv('/content/finaldataset.csv')

df1.dtypes

plt.figure(figsize=(26,5))
df1[h].boxplot()
plt.show()

# Ensure at least two classes are present
#if df['Class'].nunique() < 2:
#    raise ValueError("Target variable has only one class. Cannot train classifier.")

# Check for missing values in 'Class' column
print(df1['Class'].isnull().sum())

"""#Normalization"""

#df1.describe()

#numeric=df1.select_dtypes(include=['int64','float64']).columns
#print("numeric vlues",numeric)

#if "Class" in numeric:
#  numeric=numeric.drop("Class")

#print(df1[numeric].skew())

#from sklearn.preprocessing import StandardScaler
#Nor=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#      'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class']
#Nor=['Amount']
#scaler=StandardScaler()
#df[Nor]=scaler.fit_transform(df[Nor])

#print(df1[Nor].skew())

#log transformation
#import numpy as np
#for col in Nor:
#  df1[f'{col}_log']=np.log1p(df[col])
#pd.DataFrame({'Before':df1[Nor].skew(),'After':df1[[f'{col}_log' for col in Nor]].skew()})

#create X1 and Y1
X1 = df1.drop('Class',axis=1) # Select all columns except the last one

Y1 = df1['Class'] # Select only the last column

print(X1)

print(Y1)

# Feature scaling
from sklearn.preprocessing import StandardScaler # Import StandardScaler
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

print(Y1.unique())
print(Y1.value_counts())

df_dropna = df.dropna()
print(df_dropna)

# Balance classes using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)

X_resampled, y_resampled = smote.fit_resample(X1_scaled,Y1)

print("orignal dataset size",len(Y1))
print("resampled dataset size",len(y_resampled))

from collections import Counter
#before smote
print("before smote",Counter(Y1))
#after smote
print("after smote",Counter(y_resampled))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.2,random_state=101,stratify=Y1)

print(x_train)

print(y_train)

#logistic regration
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
train_acu=accuracy_score(y_train,model.predict(x_train))
test_acu=accuracy_score(y_test,model.predict(x_test))

print('train accuracy',train_acu)
print('test accuracy',test_acu)

model.predict(x_test)

model.predict_proba(x_test)

#sigmod function
#import math
#def sigmoid(z):
#  return 1/(1+math.exp(-z))

"""dision tree"""

#dission tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
model1=DecisionTreeClassifier(random_state=42)
model1.fit(x_train,y_train)

train_pred_dt=model1.predict(x_train)
test_pred_dt1=model1.predict(x_test)

dt_accuracy=accuracy_score(y_train,train_pred_dt)
dt_accuracy=accuracy_score(y_test,test_pred_dt1)
print("train acuracy of dt ",dt_accuracy)
print("test acuracy of dt ",dt_accuracy)

print(confusion_matrix(y_test,test_pred_dt1))

print(classification_report(y_test,test_pred_dt1))

m=['Logistic Regression','Decision Tree','Random forest','SVM']
test=[0.998999799959992,0.9988664399546576,0.9994184612131252,0.9983258731892997]
train=[0.9991331599653264,0.9988664399546576,0.9999955943254912,0.998334655035686]
bar_width=0.35
x=range(len(m))
plt.figure(figsize=(10,5))
plt.bar(x,train,width=bar_width,label='Training accuracy',color='blue')
plt.bar([i+bar_width for i in x],test,width=bar_width,label='Testing accuracy',color='green')
plt.xlabel='models'
plt.ylabel='accuracy'
plt.xticks([i+bar_width/2 for i in x],m)
plt.legend()
plt.show()

"""Radom forest

"""

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

rf_train = accuracy_score(y_train, rf.predict(x_train))
rf_test = accuracy_score(y_test, rf.predict(x_test))
print("train accuracyof rf ", rf_train)
print("test accuracyof rf ", rf_test)

cm = confusion_matrix(y_test, rf.predict(x_test))
print(cm)
print(classification_report(y_test, rf.predict(x_test)))

#svm-suport vector machine
from sklearn.svm import SVC
svm=SVC(kernel='rbf',C=1.0,gamma='scale',random_state=42)
svm.fit(x_train,y_train)

svm_train=accuracy_score(y_train,svm.predict(x_train))
svm_test=accuracy_score(y_test,svm.predict(x_test))
print("train accuracy of svm ",svm_train)
print("test accuracy of svm ",svm_test)

#hybrid approach-hybrid model-technical()
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

base_model=[
    ("rf",RandomForestClassifier(n_estimators=100,random_state=42)),
    ("dt",DecisionTreeClassifier(random_state=42)),
]
meta_model=LogisticRegression()
stacking=StackingClassifier(estimators=base_model,final_estimator=meta_model)

stacking.fit(x_train,y_train)

y_pred=stacking.predict(x_test)
print("accuracy is= ",accuracy_score(y_test,y_pred))
print("confusion matrix is\n= ",confusion_matrix(y_test,y_pred))
print("classification report is= ",classification_report(y_test,y_pred))

"""# ***Graph***

"""

#AUC ROC(reciver oprating chracterstic) curve
#use to evaluate the prformance of a classification model
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
y_prob=stacking.predict_proba(x_test)[:,1]
auc_score=roc_auc_score(y_test,y_prob)
print("auc score",auc_score)
fpr,tpr,thesholds=roc_curve(y_test,y_prob)
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

def fraud_Detection():
  def case1():
    print("Enter the details of the customer")
    a1=int(input("Enter Time"))
    v1=float(input("Enter V1"))
    v2=float(input("Enter V2"))
    v3=float(input("Enter V3"))
    v4=float(input("Enter V4"))
    v5=float(input("Enter V5"))
    v6=float(input("Enter V6"))
    v7=float(input("Enter V7"))
    v8=float(input("Enter V8"))
    v9=float(input("Enter V9"))
    v10=float(input("Enter V10"))
    v11=float(input("Enter V11"))
    v12=float(input("Enter V12"))
    v13=float(input("Enter V13"))
    v14=float(input("Enter V14"))
    v15=float(input("Enter V15"))
    v16=float(input("Enter V16"))
    v17=float(input("Enter V17"))
    v18=float(input("Enter V18"))
    v19=float(input("Enter V19"))
    v20=float(input("Enter V20"))
    v21=float(input("Enter V21"))
    v22=float(input("Enter V22"))
    v23=float(input("Enter V23"))
    v24=float(input("Enter V24"))
    v25=float(input("Enter V25"))
    v26=float(input("Enter V26"))
    v27=float(input("Enter V27"))
    v28=float(input("Enter V28"))
    amount=float(input("Enter Amount"))
    print("_________________________________________")
    input_features=[[a1,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,amount]]
    base_prediction=[]
    if all(val == 0 for val in input_features[0]):
          print("_______")
          print("no fraud")
          return
    base_prediction=[]
    G=stacking.predict(input_features)
    print("------------")
    print(G)
    if(G==0):
      print("No Fraud")
    else:
      print("Chance of fraud")
  case1()
fraud_Detection()
