
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import confusion_matrix,average_precision_score, recall_score
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[20]:


#2 rows were identified with blank data, after deleting those 2 rows on the csv file
df= pd.read_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\german_creditcard.csv')


# In[21]:


#Explore the data
df.shape
type(df)
df.size
df.head()
df.tail()


# In[22]:


#Check for missing values
pd.isnull(df).any()
pd.isnull(df).sum()
#Find out the type of each variable. Object means string or categorical in the below output.
df.info()


# In[28]:


#Find the number of customers who curned vs didnt
print('Count of labels')
df.groupby("Default_On_Payment").size()


# In[29]:


le = preprocessing.LabelEncoder()
le


# In[33]:


le.fit(df['Default_On_Payment'])


# In[34]:


dfn = le.transform(df['Default_On_Payment'])
dfn


# In[35]:


df['Y'] = dfn
df.info()
df.columns


# In[36]:


#Separate out independent categorical variables for conversion into numerical
x_catg = df.loc[:,('Status_Checking_Acc', 'Credit_History', 'Purposre_Credit_Taken', 'Savings_Acc', 'Years_At_Present_Employment', 
       'Marital_Status_Gender', 'Other_Debtors_Guarantors', 'Property',  'Other_Inst_Plans ', 'Housing',  'Job',  'Telephone', 'Foreign_Worker')]
x_catg.head()
x_catg.shape


# In[37]:


#Separate out independent numerical variables 
x_num = df.loc[:,('Duration_in_Months','Credit_Amount','Inst_Rt_Income', 'Current_Address_Yrs', 'Age','Num_CC','Dependents')]
x_num.head()


# In[38]:


#Convert Categorical Variables to Numeric by dummy coding
x_num1 = pd.get_dummies(x_catg)
x_num1.head()


# In[39]:


#Get all x's together
x = pd.concat([x_num1,x_num],axis=1)
x.head()
x.info()


# In[42]:


#Create Dependent variable
y = df['Y']
y.head()


# In[43]:


#Create train and test with 75% and 25% split
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.25, random_state=1)
train_x.shape
test_x.shape
train_y.shape
test_y.shape
type(train_x)


# In[44]:


#Build a logistic Regression Model
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# In[45]:


log.fit(train_x, train_y)


# In[46]:


log.coef_


# In[47]:


#Find out key predictor of churn
coeff = pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(log.coef_))],axis=1)
coeff.columns = ("Variable", "Coeff")
matrix = coeff.sort_values('Coeff',ascending = False)
matrix


# In[48]:


#Generate Model Diagnostics
classes = log.predict(test_x)
print(classes.size)
print('Positive Cases in Test Data:', test_y[test_y == 1].shape[0])
print('Negative Cases in Test Data:', test_y[test_y == 0].shape[0])
classes
classes.shape


# In[49]:


#Precision and Recall
print ('Accuracy score')
print (metrics.accuracy_score(test_y, classes))
print ('Precision/ Recall Metrics')
print (metrics.classification_report(test_y, classes))
print ('AUC')
auc = metrics.roc_auc_score(test_y,classes)
auc


# In[50]:


#ROC Chart
fpr, tpr, th = roc_curve(test_y, classes)
roc_auc = metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR Chart')
plt.plot(fpr,tpr, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[53]:


#Confusion Matrix
print ('Confusion Matrix')
cf = metrics.confusion_matrix(test_y,classes)
lbl1 = ["Predicted 0", "Predicted 1"]
lbl2 = ["True 0","True 1"]
sns.heatmap(cf, annot=True,cmap = "PRGn",fmt="d", xticklabels = lbl1, yticklabels = lbl2)
plt.show()

