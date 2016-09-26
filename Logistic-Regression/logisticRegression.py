import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab

class LogisticRegression(object):
    def __init__(self):
        self.train = pd.read_csv("train.csv")
    
        #print self.train
        self.Y = np.array(self.train["Survived"])
        self.Y=self.Y.reshape( (len(self.Y), 1) )
        dummies_sex=np.array( pd.get_dummies(self.train["Sex"]) )[:,:1]
       # dummies_embarked=np.array( pd.get_dummies(self.train["Embarked"]) )
        #dummies_Pclass=np.array( pd.get_dummies(self.train["Pclass"]) )
        #print dummies_sex
        age_mean=self.train["Age"].mean()
        #print self.train["Age"]
        self.train["Age"].replace("NaN",age_mean,inplace=True)
        age=np.array( self.train["Age"]).reshape((len(self.train["Age"]),1))
        #print age.shape
        self.train.drop( ["Name","Ticket","Cabin","Embarked","Sex","PassengerId","Survived","Age"], inplace = True, axis = 1 )
        trainarray = np.array(self.train)
        trainarray = np.hstack( (trainarray, dummies_sex) )
        #trainarray = np.hstack( (trainarray, dummies_embarked) )
        trainarray = np.hstack( (trainarray, age) )
        #trainarray = np.hstack( (trainarray, dummies_Pclass) )
        #print trainarray
        self.N = trainarray.shape[0]
        self.p = trainarray.shape[1] + 1
        self.I = np.ones( (self.N,1), dtype = "float64" )
        #print trainarray
        self.X = (trainarray - np.mean(trainarray, axis=0)) / np.std(trainarray, axis=0)
        self.X = np.hstack( (self.I, self.X) )
        #self.beta = ((np.random.rand(self.p)-0.5).reshape((self.p,1)))
        self.beta = [1.]*self.p
        self.beta = np.array(self.beta).reshape((self.p,1))
        #print self.beta
        
        
        self.test = pd.read_csv("test.csv")
        self.result=pd.DataFrame(self.test["PassengerId"])
        #print self.result
        dummies_sex_test=np.array( pd.get_dummies(self.test["Sex"]) )[:,:1]
        #dummies_embarked_test=np.array( pd.get_dummies(self.test["Embarked"]) )
        age_mean_test=self.test["Age"].mean()
        self.test["Age"].replace("NaN",age_mean_test,inplace=True)
        age_test=np.array( self.test["Age"]).reshape((len(self.test["Age"]),1))
        #print age.shape
        
        fare_mean_test=self.test["Fare"].mean()
        #print self.train["Age"]
        self.test["Fare"].replace("NaN", fare_mean_test, inplace=True)
        fare=np.array( self.test["Fare"]).reshape((len(self.test["Fare"]),1))
        
        self.test.drop( ["Fare","Name","Ticket","Cabin","Embarked","Sex","PassengerId","Age"], inplace = True, axis = 1 )
        testarray = np.array(self.test)
        testarray = np.hstack( (testarray, dummies_sex_test) )
        #testarray = np.hstack( (testarray, dummies_embarked_test) )
        testarray = np.hstack( (testarray, age_test) )
        testarray = np.hstack( (testarray, fare) )
        self.N_test = testarray.shape[0]
        self.p_test = testarray.shape[1] + 1
        self.I_test = np.ones( (self.N_test,1), dtype = "float64" )
        #print pd.DataFrame(testarray)
        #print np.std(testarray, axis=0)
        self.X_test = (testarray - np.mean(testarray, axis=0)) / np.std(testarray, axis=0)
        self.X_test = np.hstack( (self.I_test, self.X_test) )
        #print self.X_test
        #print self.test
    
        
    
    def sigmoidFunction(self, beta, X):
        #print beta
        #print "*****************************"
        #print X
        return float(1) / (1 + math.e**( -X.dot(beta) ) ) # P=N*(p+1)*(p+1)*1=N*1
    
    def gradient(self, beta, X, Y):
        P=self.sigmoidFunction(beta, X)
        return X.T.dot(P-Y)  # G=(p+1)*N*N*1=(p+1)*1
    
    def likelyHoodFunction(self,beta,X,Y):
        P=self.sigmoidFunction(beta,X)
        return -Y.T.dot(np.log(P))-(self.I-Y).T.dot(np.log(self.I-P))
    
    def gradientDecent(self, beta, X, Y, step=0.01, converge=0.00000001):
        likelyHood_iter = []
        likelyHood = self.likelyHoodFunction(beta, X, Y)
        likelyHood_iter.append([0, likelyHood])
        change=1
        print likelyHood
        i = 1
        while change>converge:
        #while math.e**(-likelyHood)<0.99:
            old_likelyHood = likelyHood
            beta=beta - ( step * self.gradient(beta.copy(), X, Y) )
            #print beta
            likelyHood = self.likelyHoodFunction(beta, X, Y)
            likelyHood_iter.append([i, likelyHood])
            change = abs(math.e**(-old_likelyHood) - math.e**(-likelyHood))
            i+=1
        print np.array(likelyHood_iter)
        print math.e**(-likelyHood)
        return beta
        
        
    def pred_values(self, beta, X, Y=None):
        pred_prob = self.sigmoidFunction(beta, X)
        #print pred_prob
        result=[1 if i>0.5 else 0 for i in pred_prob]
        self.result.insert(1,"Survived",result)
        self.result.to_csv("result.csv")
        #print self.result
        return result
        count=0
        for i in xrange(len(X)):
            if result[i] == Y[i]:
                count+=1
        print float(count) / len(X)
        
    def testLG(self):
        #print self.beta
        be=self.gradientDecent(self.beta, self.X, self.Y)
        #print be
        #self.pred_values(be, self.X[300:600], self.Y[300:600])
        #print pd.DataFrame(self.X_test)
        self.pred_values(beta=be, X=self.X_test)


a=LogisticRegression()
a.testLG()