
# coding: utf-8

# # Question 2

# ## To run this code, ensure that the version of the following are correct: python3.6, anaconda3

# In[66]:


import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

inp=[]
out=[]
testinp = []
testout = []
traininp=[]
trainout=[]
validinp=[]
validout=[]

#reading the training data to attain the theta and theta0 values for each 100 iterations
with open("train_warfarin.csv", newline ='\n') as f: #read the training data 
    
    reader =csv.reader(f)
    for row in reader:
        out += [[float(row[0])]]
        
        inp += [[float(row[1]), float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20])]]  
            
    X = np.array(inp) #convert the list to array
    Y = np.array(out)




# ### Below is to run the test given the theta and theta0 values trained.

# In[67]:


with open("train_warfarin.csv", newline ='\n') as ftrain: #read the test data
    reader =csv.reader(ftrain)
    for row in reader:
        traininp += [[float(row[1]), float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20])]]  
        trainout += [[float(row[0])]]
        
    trainX = np.array(traininp) #converting the list into array
    trainY = np.array(trainout)
    
with open("test_warfarin.csv", newline ='\n') as ftest: #read the test data
    reader =csv.reader(ftest)
    for row in reader:
        testinp += [[float(row[1]), float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20])]]  
        testout += [[float(row[0])]]
        
    testX = np.array(testinp) #converting the list into array
    testY = np.array(testout)

with open("validation_warfarin.csv", newline ='\n') as fvalid: #read the test data
    reader =csv.reader(fvalid)
    for row in reader:
        validinp += [[float(row[1]), float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20])]]  
        validout += [[float(row[0])]]
        
    validX = np.array(validinp) #converting the list into array
    validY = np.array(validout)

    
    


# ### Stochasticgrad function is to generate the 100th theta and theta0 in the 10000 iterations

# In[68]:


def stochasticgrad(X,Y):
    theta = np.zeros(len(X[0])) # initialise
    theta0 = 0 #initialise
    iterations = 10000
    LR = 0.1 #learning rate
    thetalist = np.zeros((100,len(X[0])))
    theta0list = np.zeros(100)
    count = 1
    
    for iteration in range(iterations): 
        
        i = random.randint(0,len(Y)-1) #generate random datapoint
        theta += (LR * (Y[i]-(np.dot(X[i],theta))-theta0) * X[i]) #updating the theta
        theta0 += (LR* (Y[i]-(np.dot(X[i],theta))-theta0)) #updating the theta0
        
        count += 1
        
        if count%100 == 0: # if the modulus is 0, suggest that this is the 100th set
            
            j = int(count/100-1)
            #store the 100th set theta and theta0
            thetalist[j]=theta 
            theta0list[j]=theta0
    
    thetalist1 = list(thetalist)
    theta0list1 = list(theta0list)
    return (thetalist1,theta0list1)
   

A= (stochasticgrad(X,Y))
#print (A)


# ### Train function is to increase test the theta and theta0 values with the three data sets (train - training set, test - test set and valid - validation set)

# In[69]:


def train(tX,tY,thetalist1,theta0list1):
    MSElist = []
    
    for j in range(len(thetalist1)):
        MSE = 0
        for i in range(len(tY)):
            #add all the MSE value of the i element in thetalist and theta0list 
            MSE += (tY[i] -(np.dot(tX[i],thetalist1[j]))-theta0list1[j])**2
        #divide the sum of MSE value by the total number of elements in the dataset (100)
        avgMSE = float(MSE/len(tY))
        #print (avgMSE)
            
        MSElist.append(avgMSE)
    return (MSElist)
        
trainY = (train(trainX,trainY,A[0],A[1])) 
testY = (train(testX,testY,A[0],A[1])) 
validY = (train(validX,validY,A[0],A[1])) 


# ### Below is to plot the graph of the three data set

# In[70]:


xaxis = list(range(100,10100,100))
trainyaxis = trainY
testyaxis = testY
validyaxis = validY

plt.plot(xaxis,trainyaxis,'ro',label='training set')
plt.plot(xaxis,testyaxis,'bo',label="test set")
plt.plot(xaxis,validyaxis,'go',label='validation set')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Mean Square Error (MSE)')
plt.xlabel('Number of iterations')
plt.show()


# ## Part b

# ### In the training set, save the value of theta and theta0 for every 100th iterations. 
# ### Using the values attained from the training set, apply on the validation set by running the stochastic gradient descent algorithm. 
# ### Calculate the Mean Square Error(MSE) for each theta and theta0
# ### The theta and theta0 value that gives the lowest MSE will be selected to be applied on the test set using stochastic gradient descent algorithm 
