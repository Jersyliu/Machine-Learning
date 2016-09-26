import numpy as np
import scipy.misc as smp
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab

class Nodes(object):
    def __init__(self,dis,index,label):
        self.dis=dis
        self.index=index
        self.label=label
        

class Solution(object):
    def __init__(self):
        self.train=pd.read_csv("train.csv")
        self.label=self.train["label"]
        self.sampleNum=len(self.label)
        del self.train["label"]
        self.colorInverse=255*np.ones((28,28))
        self.each=[]
        for i in xrange(len(self.label)):
            k=np.reshape(self.train.iloc[i],(28,28))
            self.each.append(self.colorInverse-k)
        #self.priorProbability=self.priorProbabilityOfClasses()
        #for i in self.priorProbability:
           # print str(i)+":"+str(self.priorProbability[i])
    
    def drawZeroToNine(self):
        draw=np.zeros((56,140))
        for i in xrange(10):
            index=0
            while i!=self.label[index]:
                index+=1
            rowFrom=28*(i//5)
            rowTo=28*((i//5)+1)
            colFrom=28*(i%5)
            colTo=28*((i%5)+1)
            draw[rowFrom:rowTo,colFrom:colTo]+=self.each[index]
        img=smp.toimage(draw)
        img.save("ZeroToNine.png")
        
    def priorProbabilityOfClasses(self):
        count=[0]*10
        for i in xrange(self.sampleNum):
            count[self.label[i]]+=1
        return count
        #for k in xrange(10):
            #count[k]=count[k]*1.0/self.sampleNum
        #return count
        
    def drawHistogram(self):
        plt.hist(self.label, bins=10,range=(0.0,10.0),normed=True)
        plt.show()
        
        #plt.hist(self.label, bins=20,range=(0.0,10.0), normed=1, facecolor='red', alpha=0.2)
        #plt.show()
        
    def tryOneNearestNeighbor(self):
        result=""
        for i in xrange(10):
            index=0
            while i!=self.label[index]:
                index+=1
            min=[self.L2Distance(self.each[index],self.each[index+1]),index+1]
            for j in xrange(self.sampleNum):
                dis=self.L2Distance(self.each[index],self.each[j])
                if dis>=0 and dis<min[0]:
                    min[0]=dis
                    min[1]=j
            if self.label[index]==self.label[min[1]]:
                #print "sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+"\n"
                result+=("sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+"\n")
            else:
                #print "sample is "+str(self.label[index])+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+" **\n"
                result+=("sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+" **\n")
        print "\n"+result
        return
    
    def L2Distance(self,choose,other):
        diff=choose-other
        dis=0
        for i in xrange(28):
            for j in xrange(28):
                if diff[i][j]!=0:
                    dis+=diff[i][j]**2
        if dis==0:
            return -1
        return math.sqrt(dis)
        
    def onlyZeroAndOne(self):
        count={0:[[],0],1:[[],0]}
        for i in xrange(self.sampleNum):
            if self.label[i] in count:
                count[self.label[i]][0].append(i)
                count[self.label[i]][1]+=1
        #gen1=(count[0][1])*(count[0][1]-1)/2
        #gen2=(count[1][1])*(count[1][1]-1)/2
        #imp=(count[1][1])*(count[1][1]-1)/2
        gen1=(100)*(99)/2
        gen2=(100)*(99)/2
        gen=gen1+gen2
        imp=100*100
        genuine=[0]*gen
        impostor=[0]*imp
        kg=0
        ki=0
        #file_object=open("gen.txt","w")
        #file_object2=open("imp.txt","w")
        for i in xrange(count[0][1]):
            for j in xrange(i+1,count[0][1]):
                genuine[kg]=self.L2Distance(self.each[count[0][0][i]],self.each[count[0][0][j]])
                #file_object.write(str(self.L2Distance(self.each[count[0][0][i]],self.each[count[0][0][j]]))+",")
                kg+=1
            for k in xrange(count[1][1]):
                impostor[ki]=self.L2Distance(self.each[count[0][0][i]],self.each[count[1][0][k]])
                #file_object2.write(str(self.L2Distance(self.each[count[0][0][i]],self.each[count[1][0][k]]))+",")
                ki+=1
            #print i,
        #print "\n******************\n"
        for i in xrange(count[1][1]):
            for j in xrange(i+1,count[1][1]):
                genuine[kg]=self.L2Distance(self.each[count[1][0][i]],self.each[count[1][0][j]])
                #file_object.write(str(self.L2Distance(self.each[count[1][0][i]],self.each[count[1][0][j]]))+",")
                kg+=1
           # print i,
        #print ki,kg
        #file_object.close()
        #file_object2.close()
        
        plt.hist(genuine,bins=100,facecolor='red', alpha=0.5)
        plt.hist(impostor,bins=100,facecolor='green', alpha=0.5)
        plt.show()
        
        '''
        for i in xrange(count[0][1]):
            for j in xrange(i,count[0][1]):
                genuine[kg]=self.L2Distance(self.each[count[0][0][i]],self.each[count[0][0][j]])
                kg+=1
            for k in xrange(count[1][1]):
                impostor[ki]=self.L2Distance(self.each[count[0][0][i]],self.each[count[1][0][k]])
                ki+=1
            print i,
        print "\n******************\n"
        for i in xrange(count[1][1]):
            for j in xrange(i,count[0][1]):
                genuine[kg]=self.L2Distance(self.each[count[1][0][i]],self.each[count[1][0][j]])
                kg+=1
            print i,
        print ki,kg
        '''
    def draw(self):
        #gen=np.genfromtxt("gen.txt",delimiter=",")
        #imp=np.genfromtxt("imp.txt",delimiter=",")
        #plt.hist(gen,bins=50,facecolor='red', alpha=0.5)
        #plt.hist(imp,bins=50,facecolor='green', alpha=0.5)
        #plt.show()
        r=range(1000)
        print 3
        gen=pd.read_csv("gen_copy.csv",usecols=r)
        print 1
        #imp=pd.read_csv("imp.txt",delimiter=",",usecols=r)
        print 2
        #plt.hist(imp[0:1000],bins=100,range=(1500.0,5000.0),facecolor='red', alpha=0.5)
        plt.hist(gen[0:1000],bins=100,range=(1500.0,5000.0),facecolor='green', alpha=0.5)
        plt.show()
        
        
        
    '''    
    def knn(self,choose,k):
        head=Nodes(-1)
        current=Nodes(-1)
        kvalue=[]
        for i in xrange(k):
            kvalue.append(self.L2Distance(choose,self.each[i]))
        kvalue.sort(reverse=True)
        for i in xrange(k):
            a=Nodes(kvalue[i])
            if i==0:
                head.next=a
                head=head.next
                current=a
                continue
            current.next=a
            current=current.next
        while head!=None:
            print head.value
            head=head.next
        return 
     '''
    
    def knn(self,choose,k):
        kvalue=[]
        for i in xrange(k):
            kvalue.append(Nodes(self.L2Distance(choose,self.each[i]),i,self.label[i]))
        self.buildMaxheap(kvalue)
        
        for i in xrange(self.sampleNum):
            dis=self.L2Distance(choose,self.each[i])
            if dis<kvalue[0].dis:
                kvalue[0].dis=dis
                kvalue[0].index=i
                kvalue[0].label=self.label[i]
                self.maxHeapify(kvalue,0)
        result="The "+str(k)+" votes are: "
        votes=[0]*10
        for i in xrange(k):
            result+=str(kvalue[i].label)+" "
            votes[kvalue[i].label]+=1
        result+=" Therefore, KNN result is "+str(votes.index(max(votes)))
        print result
        return
    
    def maxHeapify(self,kvalue,i):
        heapSize=len(kvalue)
        leftChild=2*i+1
        rightChild=2*i+2
        if leftChild<heapSize and kvalue[i].dis<kvalue[leftChild].dis:
            largest=leftChild
        else:
            largest=i
        if rightChild<heapSize and kvalue[largest].dis<kvalue[rightChild].dis:
            largest=rightChild
        if largest!=i:
            temp=kvalue[i]
            kvalue[i]=kvalue[largest]
            kvalue[largest]=temp
            self.maxHeapify(kvalue,largest)
    
    def buildMaxheap(self,kvaule):
        heapSize=len(kvaule)
        for i in xrange((heapSize-1)//2,-1,-1):
            self.maxHeapify(kvaule,i)
        return
        
    
    def testKnn(self):
        for i in xrange(10):
            index=0
            while i!=self.label[index]:
                index+=1
            print "The label is "+str(i)
            self.knn(self.each[index],5)
            
            
    #def threeFoldsCrossValidation(self):
        
        
        
        
        '''
        result=""
        for i in xrange(10):
            index=0
            while i!=self.label[index]:
                index+=1
            min=[self.L2Distance(self.each[index],self.each[index+1]),index+1]
            for j in xrange(self.sampleNum):
                dis=self.L2Distance(self.each[index],self.each[j])
                if dis>=0 and dis<min[0]:
                    min[0]=dis
                    min[1]=j
            if self.label[index]==self.label[min[1]]:
                #print "sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+"\n"
                result+=("sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+"\n")
            else:
                #print "sample is "+str(self.label[index])+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+" **\n"
                result+=("sample is "+str(self.label[index])+" whose index is "+str(index)+" nearest neighbor is "+str(self.label[min[1]])+" whose index is "+str(min[1])+" **\n")
        print "\n"+result
        return
'''






a=Solution()
#a.drawZeroToNine()
#a.drawHistogram()
#a.tryOneNearestNeighbor()
a.onlyZeroAndOne()
#a.draw()
#a.testKnn()