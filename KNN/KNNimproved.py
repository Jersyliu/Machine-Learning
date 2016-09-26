import numpy as np
import scipy.misc as smp
import pandas as pd
import matplotlib.pyplot as plt
import random
        
class Solution(object):
    def __init__(self):
        self.train=pd.read_csv("train.csv")
        self.label=self.train["label"]
        self.sampleNum=len(self.label)
        del self.train["label"]
        self.arraytrain=np.matrix(self.train)
        self.seperateSamples=[[] for i in xrange(10)]
        for i in xrange(self.sampleNum):
            self.seperateSamples[self.label[i]].append(self.arraytrain[i])
        for i in xrange(10):
            self.seperateSamples[i]=np.matrix(self.seperateSamples[i])
        
        #self.priorProbability=self.priorProbabilityOfClasses()
        #for i in self.priorProbability:
           # print str(i)+":"+str(self.priorProbability[i])
    '''
    def drawZeroToNine(self):
        colorInverse=255*np.ones((28,28))
        each=[]
        for i in xrange(len(self.label)):
            k=np.reshape(self.train.iloc[i],(28,28))
            each.append(colorInverse-k)
        draw=np.zeros((56,140))
        for i in xrange(10):
            index=0
            while i!=self.label[index]:
                index+=1
            rowFrom=28*(i//5)
            rowTo=28*((i//5)+1)
            colFrom=28*(i%5)
            colTo=28*((i%5)+1)
            draw[rowFrom:rowTo,colFrom:colTo]+=each[index]
        img=smp.toimage(draw)
        img.save("ZeroToNine2.png")
    '''
    '''
    def priorProbabilityOfClasses(self):
        count=[0]*10
        for i in xrange(self.sampleNum):
            count[self.label[i]]+=1
        return count
        #for k in xrange(10):
            #count[k]=count[k]*1.0/self.sampleNum
        #return count
    '''
    '''
    def drawHistogram(self):
        plt.hist(self.label, bins=10,range=(0.0,10.0),normed=True)
        plt.show()
    '''    
        
    def L2Distance(self,choose,other):
        diff=choose-other
        dis=sum(diff**2)
        return dis
    '''   
    def knn(self,choose,k):
        kvalue=[float("inf")]*k
        kindex=range(k)
        for i in xrange(self.sampleNum):
            dis=self.L2Distance(choose,self.arraytrain[i])
            m=max(kvalue)
            if dis<m:
                kindex[kvalue.index(m)]=i
                kvalue[kvalue.index(m)]=dis
        result="The "+str(k)+" votes are: "
        votes=[0]*k
        for i in xrange(k):
            result+=str(self.label[kindex[i]])+" "
            votes[self.label[kindex[i]]]+=1
        result+=" Therefore, KNN result is "+str(votes.index(max(votes)))
        print result
        return
    ''' 
    
    def knn(self,choose,k):
        kvalue=[float("inf")]*k
        kvote=[self.label[random.randint(0,k-1)] for i in xrange(k)]
        count=0
        #while (self.most(kvote)[1] < (k-1) or len(set(kvote))>2) and count<self.sampleNum:
        while len(set(kvote))!=1 and count<self.sampleNum:
            other=random.randint(0,self.sampleNum-1)
            #print other
            dis=self.L2Distance(choose,self.arraytrain[other])
            m=max(kvalue)
            if dis<m:
                kvote[kvalue.index(m)]=self.label[other]
                kvalue[kvalue.index(m)]=dis
            count+=1
        #print "the votes are: ",kvote
        #print "result is",self.most(kvote)[0]
        return self.most(kvote)[0]
        
    def most(self,kvote):
        result=[0]*10
        for i in kvote:
            result[i]+=1
        m=max(result)
        return (result.index(m),m)

    def testKnn(self):
        for j in xrange(10):
            count=0
            for i in self.seperateSamples[j][:100]:
                #print "The label is "+str(i)
                #print count,
                r=self.knn(i,11)
                if r==j:
                    count+=1
            print float(count)/len(self.seperateSamples[j][:100])
        
    
    
a=Solution()
#a.drawZeroToNine()
#a.testKnn()
#a.test2()