#!/usr/bin/env python
# coding: utf-8

# In[106]:


# Import panda, sklearn and read csv file
import pandas as pd
from pandas import *
from sklearn.tree import DecisionTreeClassifier as DT
# We can split our dataset and train model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
from sklearn import *


# Returns dataframe
musicDf = pd.read_csv('music.csv')
# 0 = FEMALE
# 1 = MALE
musicDf


# In[16]:


'''
We Clean the Data/ Prepare it
Seperate the data into two sets.The input set and output set.

Input set = age and gender
output set = genre. 
We will tell our model if a person is a male and if their ages are between 20-25 they like hip-hop
If male is between 26 -30 they like jazz. If male is above 30 they like classical music. Similar thing for women
'''
# We create the input set which will be used to train by specifying the columns we want to drop
X = musicDf.drop( columns= ['genre'])

# We create output set which will be used to test
Y = musicDf.drop(columns = ['age', 'gender'])


# In[21]:


'''
Create our model using an algorithm FROM SCIKIT-Learn
We will be using a decision tree which will look for patterns in data set
'''

# Create instance of Decision tree
model = DT()

# Takes two input, input set and ouput set
model.fit(X,Y)

# We tell the model to predict which takes a 2d array.  We will pass in a new input set
predictions = model.predict([
    [21,1],
    [22,0]
])

# Inspects prediction
predictions


# In[99]:


# We should use 70-80 % of data to train and 20-30% to train model
# We now train model using an entire dataset using train_test_split
# train_Test_split(FIRST SET, SECOND SET, What size to test). Returns tupel

# We train and test both the input and output set
X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size =0.1)

# pass training dataset
model.fit(X_train,Y_train)

# Ideally we want to create a persistent models so we wont have to train it every time we run the program
joblib.dump(model,'musicGuesser.joblib')

# now we can load our model using joblib.load
model2 = joblib.load('')


# In[112]:


# VISUALIZES DATA TREE
tree.export_graphviz(model, out_file="MuiscGuesserTree.dot",
                    feature_names = ['age','gender'],
                    label='all',
                    rounded =True,
                    filled = True)


# In[ ]:




