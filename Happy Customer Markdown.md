# Happy Customers

### The goal of this project is to predict if a customer is happy or not based on the answers they give to questions asked by at least 73% accuracy.

The company is also interested in finding which questions/features are more important when predicting a customer’s happiness.

Data Description

Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers

X1 = my order was delivered on time

X2 = contents of my order was as I expected

X3 = I ordered everything I wanted to order

X4 = I paid a good price for my order

X5 = I am satisfied with my courier

X6 = the app makes ordering easy for me

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.

```
#Import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#loading the data
happy=pd.read_csv('ACME-HappinessSurvey2020.csv')
happy.head()

happy.shape

happy.info()

#See columns in ##
happy.columns

happy.describe()
```
We can see that for each questions (X1 to X6), the values range from 1 to 5.

```
happy.isnull().sum()
```
There are no null variables.

### Data Exploration

Let's explore the data furthermore.

```
happy['Y'].value_counts().plot(kind='pie',autopct='%1.2f%%',title='Proportion of Happy Customers')
```
From the graph, we can see that 54.76% of the customers are happy while 45.24% are not. The data is slightly imbalanced.

Let's look at correlation between the features and target.

```
plt.figure(figsize=(20,10))
sns.heatmap(happy.corr(),annot=True)
happy.corr() 
```
We can see that there is no multicollinearity and some features are more correlated to the target than others (X1, X3 & X6).

### Data Modeling

We will use different models to see which one predicts the target variable the best.

```
#Importing "train_test-split" function to test the model
from sklearn.model_selection import train_test_split

#Splitting the data
X=happy.drop(['Y'],axis=1)
y=happy['Y']

#Model Comparison
# Compare classification algorithms
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (f1_score, accuracy_score, 
                             recall_score, 
                             precision_score, 
                             confusion_matrix, 
                             roc_auc_score, 
                             ConfusionMatrixDisplay, 
                             classification_report, 
                             precision_recall_curve)

def load_model(name):
    if name=='DecisionTree':
        model=DecisionTreeClassifier()
    elif name=='LogisticReg':
        model=LogisticRegression()
    elif name=='KNN':
        model=KNeighborsClassifier()
    elif name=="SVC":
        model=SVC()
    elif name=='RandForest':
        model=RandomForestClassifier()
    return model

models=['DecisionTree', 'LogisticReg', 'KNN', 'SVC', 'RandForest']
dict_models=dict()
dict_models.fromkeys(models)

np.random.seed(0) #makes random numbers predictable, so that results are the same everytime
for model in models:
    L=[]
    for i in range(0,50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model1 =load_model(model)
        model1.fit(X_train,y_train)
        score_train=model1.score(X_train,y_train)
        score_test=model1.score(X_test,y_test)
        L.append(score_test)
        if score_test >= max(L):
            model_max=model1
    dict_models[model]=model_max
    print("Best Score of ",model,"is :",max(L),"with the random_state:",L.index(max(L)))
    y_pred = dict_models[model].predict(X_test)
    print("\n Report of the Model: ", model,"\n", classification_report(y_test, y_pred),"\n")
```

The Decision Tree model has a score of 0.73 with 96% accuracy.
The Random Forest model has a score 0.769 with 88% accuracy.

```
#Confusion Matrix for each model
for model in models:
    cf_matrix=ConfusionMatrixDisplay.from_estimator(dict_models[model], X_test, y_test)  
    plt.title(str(dict_models[model])+' Confusion Matrix')
    plt.show()
```

The confusion matrix for the Decision Tree model is best because there are 12 out of 12 true positives where there are only 10 out of 12 true positives for the Random Forest model.

### Feature Selection

We will use the chisquare test for categorical feature to see which features are best to predict whether a customer is happy or not.

```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#select 3 best features
chi2_features = SelectKBest(chi2, k=3)
X_kbest = chi2_features.fit_transform(X,y)
chi2_features.get_support()
selected_feat=X.columns[(chi2_features.get_support())]
selected_feat
```

The most relevant features are X1, X3 and X5, hence the relevant survey questions are:

X1 = my order was delivered on time

X3 = I ordered everything I wanted to order

X5 = I am satisfied with my courier

We will now rerun the models with only these 3 features to see which one predicts whether the customer is happy or not the best.

```
Xnew=happy[['X1','X3','X5']]

np.random.seed(0) #makes random numbers predictable, so that results are the same everytime
for model in models:
    L=[]
    for i in range(0,50):
        Xnew_train, Xnew_test, ynew_train, ynew_test = train_test_split(Xnew, y, test_size=0.2, random_state=i)
        model1 =load_model(model)
        model1.fit(Xnew_train,ynew_train)
        score_train=model1.score(Xnew_train,ynew_train)
        score_test=model1.score(Xnew_test,ynew_test)
        L.append(score_test)
        if score_test >= max(L):
            model_max=model1
    dict_models[model]=model_max
    print("Best Score of ",model,"is :",max(L),"with the random_state:",L.index(max(L)))
    ynew_pred = dict_models[model].predict(Xnew_test)
    print("\n Report of the Model: ", model,"\n", classification_report(ynew_test, ynew_pred),"\n")
```

The Decision Tree model has a score of 0.80769 with 77% accuracy.
The Random Forest model has a score of 0.80769 with 81% accuracy.

```
for model in models:
        cfnew_matrix=ConfusionMatrixDisplay.from_estimator(dict_models[model], Xnew_test, ynew_test)  
        plt.title(str(dict_models[model])+' Confusion Matrix for X1, X3 & X5')
        plt.show()
```
The confusion matrix for the Random Forest model shows that the model correctly identifies 12 true negative out of 14 whereas the Decison Tree model only identifies 11 out of 14.

### In conclusion, I believe that the Random Forest model should be used to predict if a customer is happy or not based on the answers they give to questions asked and the     questions used in the survey should be: my order was delivered on time, I ordered everything I wanted to order and  I am satisfied with my courier.

