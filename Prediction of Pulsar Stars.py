#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[2]:


#importing the dataset
df = pd.read_csv('/Users/sadegh/Desktop/DataSet GitHub/Naive Bayes/pulsar_stars.csv')


# In[3]:


df.head()


# In[37]:


df.info()


# In[72]:


plt.figure(figsize=(12,8))
sns.heatmap(df.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set3"))
plt.title("Data summary")
plt.show()


# In[20]:


#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[74]:


plt.figure(figsize=(12,6))
plt.pie(df["target_class"].value_counts().values,
        labels=["not pulsar stars","pulsar stars"],
        autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.subplots_adjust(wspace = .2)
plt.title("Proportion of target variable in dataset")
plt.show()


# In[66]:


#Renaming columns
df = df.rename(columns={' Mean of the integrated profile':"mean_profile",
       ' Standard deviation of the integrated profile':"std_profile",
       ' Excess kurtosis of the integrated profile':"kurtosis_profile",
       ' Skewness of the integrated profile':"skewness_profile", 
        ' Mean of the DM-SNR curve':"mean_dmsnr_curve",
       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",
       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dmsnr_curve",
       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",
       })


# In[67]:


data.columns


# In[75]:


sns.pairplot(data=df,
             palette="hls",
             hue="target_class",
             vars=["mean_profile",
                   "std_profile",
                   "kurtosis_profile",
                   "skewness_profile",
                   "mean_dmsnr_curve",
                   "std_dmsnr_curve",
                  "kurtosis_dmsnr_curve"])

plt.tight_layout()
plt.show()


# In[4]:


X = df.drop('target_class',axis=1)
y = df['target_class']


# In[5]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# In[8]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[10]:


#fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


# In[11]:


y_pred = classifier.predict(X_test)


# In[12]:


from sklearn.metrics import confusion_matrix,classification_report


# In[13]:


cm = confusion_matrix(y_test,y_pred)


# In[14]:


cm


# In[30]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="OrRd" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[17]:


print(classification_report(y_test,y_pred))


# In[77]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




