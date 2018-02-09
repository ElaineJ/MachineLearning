
# coding: utf-8

# ## Question 2

# #### To run this code, ensure that the version of the following are correct: python3.6, anaconda3 


import csv
import numpy as np

inp=[]
out=[]
testinp=[]
testout=[]


with open("train_1_5.csv", newline ='\n') as f: #read the training data 
    
    reader =csv.reader(f)
    
    for row in reader:
        inp += [[float(row[0]), float(row[1])]] #obtain the input values which are of colm 1 and 2 as the vector X
        out += [[float(row[2])]] #obtain the label which are of colm 3 as Y
        
    X = np.array(inp) #convert the list to array
    Y = np.array(out)
 

with open("test_1_5.csv", newline ='\n') as ftest: #read the test data
    reader =csv.reader(ftest)
    for row in reader:
        testinp += [[float(row[0]), float(row[1])]]
        testout += [[float(row[2])]]
        
    testX = np.array(testinp)
    testY = np.array(testout)
    
    #print (testX)


def perceptron(X,Y,iterations):
    theta = np.zeros(len(X[0])) #initialise zero vector
    theta0 =0 
    
    
    #print (theta)
    for iteration in range(iterations):
        
        for i,x in enumerate(X):
            
            #when the signs are different then the output will be smaller or equal to zero
            
            if (((np.dot(X[i],theta))+theta0)*Y[i]) <=0 : #there is classification error
               
                theta0 = theta0 +Y[i] #update the theta0 value each time there is an error by the y(i)
                theta = theta + X[i]*Y[i] #update the theta value each time there is an error by the y(i)
   
    return (theta,theta0)
    
    
#interations 5
theta_5 = (perceptron(X,Y,5))[0] #theta 
theta0_5 = (perceptron(X,Y,5))[1] #theta0

#iterations 10
theta_10 = (perceptron(X,Y,10))[0] #theta
theta0_10 = (perceptron(X,Y,10))[1] #theta0



def perceptronTest(testX,testY,theta,theta0):
    numerror = 0
    numentries =  len(testY) #total number of entries in Test data set
    for i,x in enumerate(testX):
        if (((np.dot(testX[i],theta)) +(theta0))*testY[i]) <= 0: #there is classification error
            numerror += 1 #the number of data sets that had classification error
            
    accuracy = (1- (numerror / numentries)) * 100
    
    return accuracy

print ('For 5 iterations, accuracy: '+ str(perceptronTest(testX,testY,theta_5,theta0_5)) + '%' )
print ('For 10 iterations, accuracy: '+ str(perceptronTest(testX,testY,theta_10,theta0_10)) + '%')

