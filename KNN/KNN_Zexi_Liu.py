import numpy as np
import scipy.misc as smp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import time
import bisect 
class Solution(object):
    def __init__(self):
        #Read training data
        self.train = pd.read_csv("train.csv")
        self.label = self.train["label"]
        self.sampleNum = len(self.label)
        del self.train["label"]
        self.arraytrain = np.array(self.train)
        
        #Seperate digits
        self.seperateSamples = [[] for i in xrange(10)]
        for i in xrange(self.sampleNum):
            self.seperateSamples[self.label[i]].append(self.arraytrain[i])
        for i in xrange(10):
            self.seperateSamples[i] = np.array(self.seperateSamples[i])
            
        #Read test data
        self.test = pd.read_csv("test.csv")
        self.arraytest = np.array(self.test)
        
    def drawZeroToNine(self):
        draw = np.zeros((56,140))
        inverse = np.ones((56,140))
        for i in xrange(10):
            rowFrom=28*(i//5)
            rowTo=28*((i//5)+1)
            colFrom=28*(i%5)
            colTo=28*((i%5)+1)
            resha = self.seperateSamples[i][0][:]
            resha = resha.reshape((28,28))
            draw[rowFrom:rowTo,colFrom:colTo] += resha
        img = smp.toimage(inverse-draw)
        img.save("ZeroToNine.png")
    
    def onlyOneNearestNeighbor(self):
        for i in xrange(10):
            result = self.KNNpredict(self.seperateSamples[i][0],self.arraytrain,self.label,1)
            if result[0] == i:
                print "Sample is "+str(i)+", result is "+str(result[0])
                continue
            else:
                print "Sample is "+str(i)+", result is "+str(result[0])+" *"
        return 
            
    
    '''
    def L2Distance(self,Test,Train):
        TestSquareSum = np.square(Test).sum(axis=1).reshape((len(Test),1)) #add to column
        TrainSquareSum = np.square(Train).sum(axis=1).reshape((len(Train),1)) #add to row
        TestTrainIntersect = np.dot(Test,Train.T)
        DistTestTrain = -2*TestTrainIntersect+TestSquareSum+TrainSquareSum.T
        return np.array(DistTestTrain) #Return the distance matrix
    '''  
    def KNNpredict(self,Test,Train,trainlabel,k):
        result = []
        #DistTestTrain = self.L2Distance(Test,Train)
        DistTestTrain = euclidean_distances(Test,Train)
        for i in xrange(DistTestTrain.shape[0]):
            nearestKNeighborIndex = np.argsort(np.squeeze(DistTestTrain[i]))[:k]
            label = [trainlabel[j] for j in nearestKNeighborIndex]
            result.append(self.most(label))
        '''
        listresult = list(result)
        with open('submission.csv', 'w') as f:
            f.write('ImageId,Label\n')        
            for i in xrange(len(listresult)):
                f.write('%d,%d' % (i+1, listresult[i]))
                f.write('\n')
        '''
        return result
                     
    def most(self,label):
        result = [0]*10
        for i in label:
            result[i] += 1
        m=max(result)
        return result.index(m)
        
        
    def threeFoldsCrossValidation(self,k):
        print "This is "+str(k)+" fold: "
        result=0.0
        for i in xrange(3):
            sampleNum = self.sampleNum
            trainingDataNum = sampleNum//3
            arraytrain = self.arraytrain[:sampleNum]
            start_time = time.time()
            test = arraytrain[i*trainingDataNum:(i+1)*trainingDataNum]
            train = np.vstack((arraytrain[:i*trainingDataNum],arraytrain[(i+1)*trainingDataNum:]))
            label = np.array(self.label[i*trainingDataNum:(i+1)*trainingDataNum])
            trainlabel = np.hstack((self.label[:i*trainingDataNum],self.label[(i+1)*trainingDataNum:]))
            result += self.testKNN(test,train,label,trainlabel,k)
            print("--- %s seconds ---" % (time.time() - start_time))
        print result/3

    def testKNN(self,test,train,label,trainlabel,k):
        result = self.KNNpredict(test,train,trainlabel,k)
        correct = 0
        for i in (result-label):
            if i == 0 :
                correct += 1
        print float(correct)/len(test)
        return float(correct)/len(test)
        
    def GenuineAndImposter_ROC(self):
        len0 = len(self.seperateSamples[0])
        len1 = len(self.seperateSamples[1])
        Genuine0 = euclidean_distances(self.seperateSamples[0],self.seperateSamples[0]).reshape((1,len0**2))
        Genuine1 = euclidean_distances(self.seperateSamples[1],self.seperateSamples[1]).reshape((1,len1**2))
        Genuine = np.hstack((Genuine0.squeeze(),Genuine1.squeeze()))
        Imposter1 = euclidean_distances(self.seperateSamples[0],self.seperateSamples[1]).reshape((1,len0*len1))
        Imposter2 = euclidean_distances(self.seperateSamples[1],self.seperateSamples[0]).reshape((1,len0*len1))
        Imposter = np.hstack((Imposter1.squeeze(),Imposter2.squeeze()))
        '''
        plt.hist(Genuine,bins=1000,facecolor='blue', alpha=0.1)
        plt.hist(Imposter,bins=1000,facecolor='red', alpha=0.1)
        plt.show()
        '''
        #ROC
        tpr = []
        fpr = []
        Genuine.sort()
        Imposter.sort()
        eer = 0
        threshold = 0
        flag = False
        for theta in xrange(0,4500):
            tempfpr = bisect.bisect_left(Imposter, theta) / float(len(Imposter))
            temptpr = bisect.bisect_left(Genuine, theta) / float(len(Genuine))
            fpr.append(tempfpr)
            tpr.append(temptpr)
            if flag == False and temptpr > 1 - tempfpr:
                eer = tempfpr
                threshold = theta
                flag = True
        print "EER is: "+str(eer)
        print "Threshold is "+str(threshold)
        '''
        plt.plot(fpr, tpr, 'go-')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC')
        plt.grid()
        plt.show()
        '''
        
    def plotConfusion_matrix(self):
        result = self.KNNpredict(self.arraytrain,self.arraytrain,self.label,5)
        CM = confusion_matrix(self.label, result)
        print(CM)
    
            
            
a = Solution()
#a.drawZeroToNine()
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))
#a.threeFoldsCrossValidation(5)
#for k in [5,7,9,11]:
    #a.threeFoldsCrossValidation(k)
#a.KNNpredict(a.arraytest,a.arraytrain,a.label,k)
#print("--- %s seconds ---" % (time.time() - start_time))
#a.GenuineAndImposter_ROC()
#a.onlyOneNearestNeighbor()
a.plotConfusion_matrix()
            