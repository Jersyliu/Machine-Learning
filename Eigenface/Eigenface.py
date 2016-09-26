import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
#%matplotlib inline
from sklearn.linear_model import LogisticRegression

class Eigenface(object):
    def __init__(self):

        self.train_labels, self.train_data = [], []
        self.test_labels, self.test_data = [], []
        
        #input training data
        for line in open("./faces/train.txt"):
            im = misc.imread(line.strip().split()[0])
            self.train_data.append(im.reshape(2500,))
            self.train_labels.append(line.strip().split()[1])
    
        self.train_data, self.train_labels = np.array(self.train_data, dtype=float), np.array(self.train_labels, dtype=int)
        '''
        print(train_data.shape, train_labels.shape)
        plt.imshow(train_data[1, :].reshape(50,50), cmap = cm.Greys_r)
        plt.show()
        '''
        
        #input testing data
        for line in open("./faces/test.txt"):
            im = misc.imread(line.strip().split()[0])
            self.test_data.append(im.reshape(2500,))
            self.test_labels.append(line.strip().split()[1])
        self.test_data, self.test_labels = np.array(self.test_data, dtype=float), np.array(self.test_labels, dtype=int)
        '''
        print(test_data.shape, test_labels.shape)
        plt.imshow(test_data[10, :].reshape(50,50), cmap = cm.Greys_r)
        plt.show()
        '''

        self.miu = self.averageFace()
        self.U_train, self.d_train, self.VH_train, self.U_test, self.d_test, self.VH_test = self.eigenFace()
        

    def averageFace(self):
        sumUp = sum(self.train_data)
        averageFace = sumUp / len(self.train_data)
        #plt.imshow(averageFace.reshape(50,50), cmap = cm.Greys_r)
        #plt.show()
        return averageFace

    def meanSubstraction(self):
        meansubstracted_train = self.train_data-self.miu
        #plt.imshow(meansubstracted_train[1].reshape(50,50), cmap = cm.Greys_r)
        #plt.show()
        meansubstracted_test = self.test_data-self.miu
        #plt.imshow(meansubstracted_test[10].reshape(50,50), cmap = cm.Greys_r)
        #plt.show()

    def eigenFace(self):
        U,d,VH = np.linalg.svd(self.train_data, full_matrices = True)
        U_t,d_t,VH_t = np.linalg.svd(self.test_data, full_matrices = True)
        return (U,d,VH,U_t,d_t,VH_t)
        #print (VH[0].reshape(50,50))
        #draw = np.zeros((50,50))
        '''
        plt.figure(figsize=(20,10))
        for i in range(10):
            
            rowFrom=50*(i//5)
            rowTo=50*((i//5)+1)
            colFrom=50*(i%5)
            colTo=50*((i%5)+1)
            draw[rowFrom:rowTo,colFrom:colTo]+= VH[i].reshape(50,50)
            
            plt.imshow(VH[i].reshape(50,50), cmap = cm.Greys_r)
            plt.show()
            
            
            plt.subplot(2,5,i+1)
            plt.imshow(VH[i].reshape(50,50), cmap = cm.Greys_r)
        plt.show()
            
            #draw = np.hstack((draw,abs(VH[i].reshape(50,50))))
        #plt.imshow(draw, cmap = cm.Greys_r)
        #plt.show()
        '''


    def lowRankApproximation(self):
        fro = []
        for r in range(1,201):
            Xhat = self.U_train[:,:r].dot(np.diag(self.d_train)[:r,:r].dot(self.VH_train[:r,:]))
            fro.append(np.linalg.norm(self.train_data-Xhat,ord = "fro"))
        plt.plot(range(1,201),fro,"go")
        plt.ylabel("Fro Norm")
        plt.xlabel("r")
        plt.show()

    def eigenfaceFeature(self,r):
        F_train = self.train_data.dot(self.VH_train[:r,:].T)
        #F_test = self.test_data.dot(self.VH_test[:r,:].T)
        F_test = self.test_data.dot(self.VH_train[:r,:].T)
        return (F_train,F_test)
        #print (F.shape)

    def FaceRecognition(self):
        
        F_train, F_test = self.eigenfaceFeature(10)
        #print (F_train.shape,F_test.shape)
        model = LogisticRegression()
        model.fit(F_train,self.train_labels)
        #result = model.predict(F_test)
        score = model.score(F_test,self.test_labels)
        print (score)
        '''
        accuracy = []
        for r in range(1,201):
            count =  0
            F_train, F_test = self.eigenfaceFeature(r)
            model = LogisticRegression()
            #print (self.train_labels)
            model.fit(F_train,self.train_labels)
            
            result = model.predict(F_test)
            print (result)
            return
            for i in range(len(result)):
                if result[i] == self.test_labels[i]:
                    count += 1
            score = float(count) / len(result)
            
            score = model.score(F_test,self.test_labels)
            accuracy.append(score)
        plt.plot(range(1,201),accuracy,"go")
        plt.ylabel("Accuracy")
        plt.xlabel("r")
        plt.show()
        '''

a = Eigenface()
#a.averageFace()
#a.meanSubstraction()
#.eigenFace()
#a.lowRankApproximation()
#a.eigenfaceFeature(10)
a.FaceRecognition()

















        
