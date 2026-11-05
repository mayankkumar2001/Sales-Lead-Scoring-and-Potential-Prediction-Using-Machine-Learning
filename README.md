# Sales Lead Status Prediction Using Machine Learning

## Project Overview
This project explores sales enquiry data and builds classification models to predict the status of a sales lead. It analyses enquiry patterns across products, sources, sales agents, locations, and delivery modes.

## Objectives
- Explore sales enquiry patterns across products, channels, agents, and locations.
- Prepare categorical and numerical data for machine learning.
- Train and evaluate multiple classification algorithms.
- Compare model performance using accuracy, precision, recall, and F1-score.

## Dataset
The notebook analyses sales enquiry data and uses fields such as product, source, sales agent, location, delivery mode, and lead status. Contact details are excluded from model features.

## Project Workflow
1. Load and inspect the enquiry dataset.
2. Perform exploratory data analysis and visualise category distributions.
3. Clean and prepare features, including categorical encoding.
4. Select model features and split data into training and testing sets.
5. Train, tune, and evaluate classification algorithms.

## Machine Learning Models
- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Bagging

## Evaluation
The notebook evaluates classification models using accuracy and classification reports, including precision, recall, and F1-score.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost

## Project Notebook
Open the project notebook: `Pro_Clt_Fi(1).ipynb`.

## Complete Project Code
The complete source code from the notebook is included below. Expand the section to view it.

<details>
<summary>View complete project code</summary>

### Code Cell 1

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


### Code Cell 2

```python
df = pd.read_csv('data.csv')
```


### Code Cell 3

```python
df.head()
```


### Code Cell 4

```python
df.tail()
```


### Code Cell 5

```python
df.info()
```


### Code Cell 6

```python
df.describe()
```


### Code Cell 7

```python
df.shape
```


### Code Cell 8

```python
df.describe(include='O')
```


### Code Cell 9

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Product_ID')
plt.xlabel('PRODUCT ID',fontsize=20)
plt.show()
```


### Code Cell 10

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Source',hue='Source')
plt.xlabel('SOURCE',fontsize=20)
plt.xticks(rotation=90)
plt.show()
```


### Code Cell 11

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Sales_Agent',hue='Sales_Agent')
plt.xlabel('Sales_Agent',fontsize=20)
plt.xticks(rotation=90)
plt.show()
```


### Code Cell 12

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Location',hue='Location')
plt.xlabel('Location',fontsize=20)
plt.xticks(rotation=45)
plt.show()
```


### Code Cell 13

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Delivery_Mode',hue='Delivery_Mode')
plt.xlabel('Delivery_Mode',fontsize=20)
plt.xticks(rotation=45)
plt.show()
```


### Code Cell 14

```python
plt.figure(figsize=(12,7))
sns.countplot(data=df,x='Status',hue='Status')
plt.xlabel('Status',fontsize=20)
plt.xticks(rotation=45)
plt.show()
```


### Code Cell 15

```python
len(df[df['Source'] == 'Call'])
```


### Code Cell 16

```python
len(df[df['Source'] == 'Call'])/965*100
```


### Code Cell 17

```python
## The total percentage of Source enquiry via call is 374 out of 965 which is 38%
```


### Code Cell 18

```python
len(df[(df['Product_ID'] == 18) & (df['Status'] == 'Potential')])
```


### Code Cell 19

```python
## The number Potential status of Product id 18 is 62
```


### Code Cell 20

```python
len(df[(df['Sales_Agent'] == 'Sales-Agent-9') & ((df['Status'] == 'Potential') | (df['Status'] == 'In Progress Positive'))])/965*100
```


### Code Cell 21

```python
### Percenatge of Potential and In Progress Positive - Status where the Sales Agent 9 has handled is 12.3%
```


### Code Cell 22

```python
len(df[(df['Sales_Agent'] == 'Sales-Agent-7') & ((df['Status'] == 'Potential') | (df['Status'] == 'In Progress Positive'))])/965*100
```


### Code Cell 23

```python
### Percenatge of Potential and In Progress Positive - Status where the Sales Agent 7 has handled is 8.39%
```


### Code Cell 24

```python
len(df[(df['Location'] == 'Bangalore') & (df['Source'] == 'Call')])/965*100
```


### Code Cell 25

```python
### Percentage of people who enquire from call via banglore are 9.32%
```


### Code Cell 26

```python
len(df[df['Product_ID'] == 18])/965*100
```


### Code Cell 27

```python
## 26 percent of the enquiry is about the Product ID 18
```


### Code Cell 28

```python
df['Status'].unique()
```


### Code Cell 29

```python
len(df[df['Status'] == 'Junk Lead'])
```


### Code Cell 30

```python
len(df[df['Status'] == 'Junk Lead'])/965*100
```


### Code Cell 31

```python
## The percentage of status with Junk Lead are 10% of the total data
```


### Code Cell 32

```python
len(df[df['Status'] == 'Just Enquiry'])
```


### Code Cell 33

```python
len(df[df['Status'] == 'Just Enquiry'])/965*100
```


### Code Cell 34

```python
## The percentage of status with Just Enquiry are 10% of the total data
```


### Code Cell 35

```python
df.head()
```


### Code Cell 36

```python
df['Product_ID'].unique()
```


### Code Cell 37

```python
df.isnull().sum()
```


### Code Cell 38

```python
## This Data contains of 4 Features which have Null Values
## Since these null values are unique and are categorical we have to Replace with the most occuring values which is calles as MODE
## We will include all the null values with the MODE
```


### Code Cell 39

```python
df['Product_ID'].mode()
```


### Code Cell 40

```python
df['Product_ID'] = df['Product_ID'].fillna(18)
```


### Code Cell 41

```python
df.isnull().sum()
```


### Code Cell 42

```python
### The NULL values of the Product ID feature has been replaced with the mode which is 18
```


### Code Cell 43

```python
df['Source'].mode()
```


### Code Cell 44

```python
df['Source'] = df['Source'].fillna(df['Source'].mode()[0])
```


### Code Cell 45

```python
df.isnull().sum()
```


### Code Cell 46

```python
df['Sales_Agent'].mode()
```


### Code Cell 47

```python
df['Sales_Agent'] = df['Sales_Agent'].fillna(df['Sales_Agent'].mode()[0])
```


### Code Cell 48

```python
df.isnull().sum()
```


### Code Cell 49

```python
df['Location'].mode()
```


### Code Cell 50

```python
df['Location'] = df['Location'].fillna(df['Location'].mode()[0])
```


### Code Cell 51

```python
df.isnull().sum()
```


### Code Cell 52

```python
df.head()
```


### Code Cell 53

```python
df["Product_ID"].value_counts()
```


### Code Cell 54

```python
df['Product_ID'] = df['Product_ID'].replace([12,2,1,3,13,11,24,21,17,26],0)
```


### Code Cell 55

```python
df["Product_ID"].value_counts()
```


### Code Cell 56

```python
df["Source"].value_counts()
```


### Code Cell 57

```python
df["Source"] = df["Source"].replace(['Live Chat-Direct','Live Chat-Google Organic','Live Chat -PPC','Live Chat-Blog','Live Chat-CPC','Live Chat-Google Ads','Live Chat-Adwords Remarketing'],'Chat')
```


### Code Cell 58

```python
df["Source"].value_counts()
```


### Code Cell 59

```python
df["Source"] = df["Source"].replace(['Customer Referral','Existing Client','By Recommendation','E-Mail Message'],'other')
df["Source"].value_counts()
```


### Code Cell 60

```python
### So as seen above stacked all the Email related unique values to the email
```


### Code Cell 61

```python
df["Location"].value_counts()
```


### Code Cell 62

```python
df["Location"] = df["Location"].replace(['UAE','Singapore'],'Asia')
```


### Code Cell 63

```python
df["Location"].value_counts()
```


### Code Cell 64

```python
df["Location"] = df["Location"].replace(['UK'],'EUROPE')
```


### Code Cell 65

```python
df["Location"].value_counts()
```


### Code Cell 66

```python
## Renamed the similar values such as combined UAE and Singapore to Asia and UK to Europe
```


### Code Cell 67

```python
df['Delivery_Mode'].value_counts()
```


### Code Cell 68

```python
### Did not make any changes in the Delivery Mode
```


### Code Cell 69

```python
df['Status'].value_counts()
```


### Code Cell 70

```python
df['Status'] = df['Status'].replace(['In Progress Positive','Potential','CONVERTED','converted','Open','Long Term'],'high_potential')
```


### Code Cell 71

```python
df['Status'] = df['Status'].replace(['Not Responding','Just Enquiry','Junk Lead','In Progress Negative','LOST'],'low_potential')
```


### Code Cell 72

```python
df['Status'].value_counts()
```


### Code Cell 73

```python
from sklearn.preprocessing import LabelEncoder
```


### Code Cell 74

```python
encoder = LabelEncoder()
```


### Code Cell 75

```python
df.head()
```


### Code Cell 76

```python
colls = ['Product_ID','Source','Sales_Agent','Location','Delivery_Mode','Status']
```


### Code Cell 77

```python
for col in colls:
    df[col] = encoder.fit_transform(df[col])
```


### Code Cell 78

```python
df.head()
```


### Code Cell 79

```python
df.drop(columns=['Mobile','EMAIL','Created'],inplace=True)
```


### Code Cell 80

```python
df.head()
```


### Code Cell 81

```python
plt.figure(figsize=(9,7))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.xticks(rotation=0)
plt.plot()
```


### Code Cell 82

```python
df.info()
```


### Code Cell 83

```python
df.head()
```


### Code Cell 84

```python
X = df.drop(columns='Status')
```


### Code Cell 85

```python
y = df['Status']
```


### Code Cell 86

```python
sns.countplot(data=df,x='Status',hue='Status')
```


### Code Cell 87

```python
df['Status'].value_counts()
```


### Code Cell 88

```python
from sklearn.model_selection import train_test_split
```


### Code Cell 89

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


### Code Cell 90

```python
X_train.shape
```


### Code Cell 91

```python
X_test.shape
```


### Code Cell 92

```python
y_train.shape
```


### Code Cell 93

```python
y_test.shape
```


### Code Cell 94

```python
from sklearn.linear_model import LogisticRegression
```


### Code Cell 95

```python
m_log = LogisticRegression()
```


### Code Cell 96

```python
m_log.fit(X_train,y_train)
```


### Code Cell 97

```python
y_pred_log_test = m_log.predict(X_test)
```


### Code Cell 98

```python
y_pred_log_train = m_log.predict(X_train)
```


### Code Cell 99

```python
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report,confusion_matrix
```


### Code Cell 100

```python
log_train_accuracy = accuracy_score(y_pred_log_train,y_train)
```


### Code Cell 101

```python
print("Training accuracy of Logistics regression model is ", log_train_accuracy*100)
print("Logistic regression training Classification Report for Training : \n",classification_report(y_pred_log_train,y_train))
```


### Code Cell 102

```python
log_test_accuracy = accuracy_score(y_pred_log_test,y_test)
print("Testing accuracy of Logistic Regression Model is ",log_test_accuracy)
print("Logistic regression training Classification Report for Training : \n",classification_report(y_pred_log_test,y_test))
```


### Code Cell 103

```python
from sklearn.neighbors import KNeighborsClassifier
```


### Code Cell 104

```python
error_rate = []
```


### Code Cell 105

```python
for i in range(2,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
print("Error Rate is ",error_rate)
```


### Code Cell 106

```python
### Plotting the error Rate :
```


### Code Cell 107

```python
plt.figure(figsize=(7,5))
plt.plot(range(2,11),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```


### Code Cell 108

```python
### Taking the K value as 5 since it has the least Errors
```


### Code Cell 109

```python
knn = KNeighborsClassifier(n_neighbors=5)
```


### Code Cell 110

```python
knn.fit(X_train,y_train)
```


### Code Cell 111

```python
y_pred_knn1 =knn.predict(X_train)
```


### Code Cell 112

```python
y_pred_knn2 =knn.predict(X_test)
```


### Code Cell 113

```python
knn_accuracy = accuracy_score(y_pred_knn1,y_train)
print("The Training accuracy of the KNN is",knn_accuracy*100)
print("The Classifiaction report of KNN for train data is \n",classification_report(y_pred_knn1,y_train))
```


### Code Cell 114

```python
knn_accuracy = accuracy_score(y_pred_knn2,y_test)
print("The Training accuracy of the KNN for test data  is",knn_accuracy*100)
print("The Classifiaction report of KNN for train data is \n",classification_report(y_pred_knn2,y_test))
```


### Code Cell 115

```python
knn1 = KNeighborsClassifier()
```


### Code Cell 116

```python
from sklearn.ensemble import BaggingClassifier
```


### Code Cell 117

```python
bagg = BaggingClassifier(estimator=knn1,n_estimators=50)
## n_estimators = number of base learners
```


### Code Cell 118

```python
bagg.fit(X_train,y_train)
```


### Code Cell 119

```python
y_pred = bagg.predict(X_train)
```


### Code Cell 120

```python
y_pred2 = bagg.predict(X_test)
```


### Code Cell 121

```python
bagg_accuracy = accuracy_score(y_pred,y_train)
```


### Code Cell 122

```python
print("The Training accuracy of the bagging for train data  is",bagg_accuracy*100)
```


### Code Cell 123

```python
bagg_accuracy = accuracy_score(y_pred2,y_test)
print("The Training accuracy of the bagging for test data  is",bagg_accuracy*100)
```


### Code Cell 124

```python
from sklearn.tree import DecisionTreeClassifier
```


### Code Cell 125

```python
dtc = DecisionTreeClassifier()
```


### Code Cell 126

```python
dtc.fit(X_train,y_train)
```


### Code Cell 127

```python
y_pred_train = dtc.predict(X_train)
```


### Code Cell 128

```python
y_pred_test = dtc.predict(X_test)
```


### Code Cell 129

```python
train_accuracy = accuracy_score(y_pred_train,y_train)
```


### Code Cell 130

```python
print("The accuracy score of Train data via Decision Tree Classifier is ",train_accuracy*100)
print("The classification of Decision Tree on Train data is ",classification_report(y_pred_train,y_train))
```


### Code Cell 131

```python
test_accuracy = accuracy_score(y_pred_test,y_test)
```


### Code Cell 132

```python
print("The accuracy score of Train data via Decision Tree Classifier is ",test_accuracy*100)
print("The classification of Decision Tree on Train data is ",classification_report(y_pred_test,y_test))
```


### Code Cell 133

```python
from sklearn.model_selection import GridSearchCV
```


### Code Cell 134

```python
params = {
    "criterion":("gini", "entropy"),
    "splitter":("best", "random"),
    "max_depth":(list(range(1, 30))),
    "min_samples_split":[2, 3, 4,5],
    "min_samples_leaf":list(range(1, 10)),
}


tree_c = DecisionTreeClassifier(random_state=3)
tree_cv = GridSearchCV(tree_c, params, scoring="f1", n_jobs=-1, verbose=2, cv=3)

tree_cv.fit(X_train,y_train)

# Get the best parameter
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params}")
```


### Code Cell 135

```python
# placing the parameter on decision tree classifier
dt1=DecisionTreeClassifier(criterion='gini',
                           max_depth=13,min_samples_leaf=1,
                           min_samples_split=2,splitter='best')
```


### Code Cell 136

```python
# fitting training data
dt1.fit(X_train,y_train)

# prdiction on testing data
yhyp = dt1.predict(X_test)
```


### Code Cell 137

```python
test_accuracy = accuracy_score(y_test,yhyp)
print("Accuracy score:",test_accuracy*100)
print("Classification report:"'\n',classification_report(y_test,yhyp))
```


### Code Cell 138

```python
from sklearn.ensemble import RandomForestClassifier
```


### Code Cell 139

```python
rf = RandomForestClassifier()
```


### Code Cell 140

```python
rf.fit(X_train,y_train)
```


### Code Cell 141

```python
y_pred_train = rf.predict(X_train)
```


### Code Cell 142

```python
y_pred_test = rf.predict(X_test)
```


### Code Cell 143

```python
rf_train_accuracy = accuracy_score(y_pred_train,y_train)
```


### Code Cell 144

```python
print("The training accuracy of Randon Forest Classifier on train data is ",rf_train_accuracy*100)
print("The classification report of Randon Forest Classifier for train data is \n",classification_report(y_pred_train,y_train))
```


### Code Cell 145

```python
rf_test_accuracy = accuracy_score(y_pred_test,y_test)
```


### Code Cell 146

```python
print("The training accuracy of Randon Forest Classifier on Test data is ",rf_test_accuracy*100)
print("The classification report of Randon Forest Classifier for test data is \n",classification_report(y_pred_test,y_test))
```


### Code Cell 147

```python
from sklearn.model_selection import RandomizedSearchCV
# not used grid search cv beacuse memory reason

n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
max_features = ['auto', 'sqrt'] ### auto - select randomly, sqrt -
max_depth = [int(x) for x in np.linspace(10, 110, num=10)] ### level in the tree
max_depth.append(None)
min_samples_split = [2, 5, 15]
min_samples_leaf = [1, 2, 4]


# make a dictionary of parameter
random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

#  Object creation
rf_clf1 = RandomForestClassifier(random_state=42)
# provide random state beacuse select row and column randomly


# randomized search cv with parameter
rf_cv = RandomizedSearchCV(estimator=rf_clf1, scoring='f1',param_distributions=random_grid, n_iter=100, cv=3,
                               verbose=3, random_state=42, n_jobs=-1)


rf_cv.fit(X_train, y_train)

# Step:6 Getting the best parameter
rf_best_params = rf_cv.best_params_
print(f"Best paramters: {rf_best_params})")
```


### Code Cell 148

```python

rf_clf2 = RandomForestClassifier(**rf_best_params)

rf_clf2.fit(X_train,y_train)

y_predict = rf_clf2.predict(X_test)
```


### Code Cell 149

```python
test_accuracy = accuracy_score(y_test,y_predict)
print("Testing accuracy after hyper parameter tunning:",test_accuracy*100)
```


### Code Cell 150

```python
import xgboost
```


### Code Cell 151

```python
from xgboost import XGBClassifier
```


### Code Cell 152

```python
xgb= XGBClassifier()
```


### Code Cell 153

```python
xgb.fit(X_train,y_train)
```


### Code Cell 154

```python
xgb_train_pred = xgb.predict(X_train)
```


### Code Cell 155

```python
xgb_test_pred = xgb.predict(X_test)
```


### Code Cell 156

```python
xgb_accuracy = accuracy_score(xgb_train_pred,y_train)
```


### Code Cell 157

```python
print("The accuracy of XGBoost on train data is ",xgb_accuracy*100)
print("The classification report of XG Boost on Train data is \n",classification_report(xgb_train_pred,y_train))
```


### Code Cell 158

```python
xgb_accuracy = accuracy_score(xgb_test_pred,y_test)
```


### Code Cell 159

```python
print("The accuracy of XGBoost on test data is ",xgb_accuracy*100)
print("The classification report of XG Boost on test data is \n",classification_report(xgb_test_pred,y_test))
```


### Code Cell 160

```python
# Step:1 import library
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.001,0.01,0.1, 0.03, 0.06, 0.1, 0.15, 0.02,0.003, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [3,5,6,7,8,10],
              'n_estimators': [50,65,80,100,130,150],
              'reg_alpha': [0,0.01,0.1,0.02,0.2,0.4,0.8,1.6,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.01,0.1,0.02,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}

# Step:2 Object creation with parameter
XGB = XGBClassifier(random_state=42)

# Step:3 Create randomized search cv with parameter
rcv = RandomizedSearchCV(estimator=XGB,scoring='f1',param_distributions=param_grid,n_iter=150,
                        cv=5,verbose=3,random_state=5,n_jobs=-1)

#estimator--number of decision tree
#scoring--->performance matrix to check performance
#param_distribution-->hyperparametes(dictionary we created)
#n_iter--->Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.default=10
##cv------> number of flods
#verbose=Controls the verbosity: the higher, the more messages.
#n_jobs---->Number of jobs to run in parallel,-1 means using all processors.

# Step:4 Fitting training data on randomized search cv
rcv.fit(X_train,y_train)

# Step:5 Get best parameters
rcv_best_parameter = rcv.best_params_
print(f"Best parameter: {rcv_best_parameter}")
```


### Code Cell 161

```python
# Step:6 Place the best parameter
XGB2 = XGBClassifier(reg_lambda=25.6,reg_alpha=0.8,n_estimators=150,max_depth=10,learning_rate=0.4,gamma=0.2)

# Step:7 Fitting the training data
XGB2.fit(X_train,y_train)

# Step:8 Prediction on testing data
XGB_prediction = XGB2.predict(X_test)
```


### Code Cell 162

```python
XGB_accuracy = accuracy_score(XGB_prediction,y_test)
print("Accuracy score after hyperparameter tunning",XGB_accuracy*100)
print("Classification report: \n",classification_report(XGB_prediction,y_test))
```


</details>

## Resume Project Title
**Sales Lead Status Prediction Using Machine Learning**

**Resume description:** Analysed sales enquiry data and developed classification models to predict lead status, applying exploratory analysis, data preparation, categorical encoding, model evaluation, and hyperparameter tuning.
