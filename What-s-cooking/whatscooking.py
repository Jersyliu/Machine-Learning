#-*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation

class WhatsCooking(object):

    def __init__(self):
        self.train = pd.read_json("train.json")
        self.test = pd.read_json("test.json")
        self.train_cuisine = self.train["cuisine"]
        self.train_id = self.train["id"]
        self.train_ingredients = self.train["ingredients"]
        self.test_id = self.test["id"]
        self.test_ingredients = self.test["ingredients"]
        self.numOfCuisine, self.numOfCuisineType, self.ingredients_sequence = self.aboutData()
        self.binaryTrain, self.binaryTest = self.binaryRepresent()
        print self.binaryTrain.shape
        
    def aboutData(self):
        howManyDish = len(self.train_cuisine)
        howManyType = len(set(self.train_cuisine))
        ingredients = {}
        for (i,key) in enumerate(self.train_ingredients):
            for j in key:
                if j not in ingredients:
                    ingredients[j] = 1
        ingredients_sequence = {}
        for (i,key) in enumerate(ingredients.keys()):
            ingredients_sequence[key] = i
        return (howManyDish,howManyType,ingredients_sequence)
        
    def binaryRepresent(self):
        binaryTrainArray = np.zeros((self.numOfCuisine, len(self.ingredients_sequence)))
        binaryTestArray = np.zeros((len(self.test_id), len(self.ingredients_sequence)))
        for (i,ingredients) in enumerate(self.train_ingredients):
            for j in ingredients:
                binaryTrainArray[i][self.ingredients_sequence[j]] = 1
        for (i,ingredients) in enumerate(self.test_ingredients):
            for j in ingredients:
                if j in self.ingredients_sequence:
                    binaryTestArray[i][self.ingredients_sequence[j]] = 1
        return (binaryTrainArray, binaryTestArray)

        
    def threeFoldCrossValidation(self):
        #GNB = GaussianNB()
        #BNB = BernoulliNB()
        #scores = cross_validation.cross_val_score(GNB, self.binaryTrain, self.train_cuisine, cv=3)
        #scores = cross_validation.cross_val_score(BNB, self.binaryTrain, self.train_cuisine, cv=3)
        LR = LogisticRegression()
        scores = cross_validation.cross_val_score(LR, self.binaryTrain, self.train_cuisine, cv=3)
        print scores
        
    def LogisticRegressionFinal(self):
        LR = LogisticRegression()
        LR.fit(self.binaryTrain, self.train_cuisine)
        result = LR.predict(self.binaryTest)
        #print result.shape
        
        with open("submission.csv", "w") as f:
            f.write("id,cuisine\n")        
            for (i,key) in enumerate(result):
                f.write("%d,%s" % (self.test_id[i], str(key)))
                f.write("\n")
        
        
    
a = WhatsCooking()
#a.aboutData()
#a.binaryRepresent()
#a.threeFoldCrossValidation()
a.LogisticRegressionFinal()