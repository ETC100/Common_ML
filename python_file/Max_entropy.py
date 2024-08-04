#usr/bin/python3


import numpy as np
import time
from collections import defaultdict


class max_entropy:
    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        self.trainDataList = trainDataList          #train data
        self.trainLabelList = trainLabelList        #train label
        self.testDataList = testDataList            #test data
        self.testLabelList = testLabelList          #test label
        self.featureNum = len(trainDataList[0])     #feature count
    
        self.N = len(trainDataList)                 #sample size
        self.n = 0                                  #total number of（xi，y）
        self.M = 10000                              #
        self.fixy = self.calc_fixy()                #the number for each（xi，y）
        self.w = [0] * self.n                       #w
        self.xy2idDict, self.id2xyDict = self.createSearchDict() #id to (x,y) (x,y) to id
        self.Ep_xy = self.calcEp_xy()               #Ep_xy expectation
        
    def calc_fixy(self):
        '''
        calculate the number of (x, y) in test data
        :return:
        '''
        fixyDict = [defaultdict(int) for i in range(self.featureNum)]
        for i in range(len(self.trainDataList)):
            for j in range(self.featureNum):
                fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1
        for i in fixyDict:
            self.n += len(i)
        return fixyDict    
        
    def createSearchDict(self):
        '''
        xy2idDict：from (x,y) to id
        id2xyDict：from id to (x,y)
        '''
        xy2idDict = [{} for i in range(self.featureNum)]
        id2xyDict = {}

        index = 0
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                xy2idDict[feature][(x, y)] = index
                id2xyDict[index] = (x, y)
                index += 1

        return xy2idDict, id2xyDict
    
    def calcEp_xy(self):
        Ep_xy = [0] * self.n
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                id = self.xy2idDict[feature][(x, y)]
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N
        return Ep_xy
    
    def calcPwy_x(self, X, y):
        numerator = 0
        Z = 0
        
        for i in range(self.featureNum):
            if (X[i], y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], y)]
                numerator += self.w[index]
            if (X[i], 1-y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[index]
        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator
        return numerator / Z
    
    def calcEpxy(self):
        Epxy = [0] * self.n
        for i in range(self.N):
            Pwxy = [0] * 2
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)
            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy
    
    def maxEntropyTrain(self, iter = 500):
        for i in range(iter):
            iterStart = time.time()
            Epxy = self.calcEpxy()
            
            sigmaList = [0] * self.n
            for j in range(self.n):
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])
    
            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]
    
            iterEnd = time.time()
            print('iter:%d:%d, time:%d'%(i, iter, iterStart - iterEnd))
            
    def predict(self, X):
        '''
        predict the value
        '''
        result = [0] * 2
        for i in range(2):
            result[i] = self.calcPwy_x(X, i)
        return result.index(max(result))

    def test(self):
        '''
        test the model
        '''
        errorCnt = 0
        for i in range(len(self.testDataList)):
            result = self.predict(self.testDataList[i])
            if result != self.testLabelList[i]:   errorCnt += 1
        return 1 - errorCnt / len(self.testDataList)
    
def loadData(fileName):
    data_list = []
    label_list = []

    with open(fileName, "r") as f:
        for line in f.readlines():
            if line.startswith("label"): continue
            curline = line.strip().split(",")
            if (int(curline[0]) >= 5):
                label_list.append(1)
            else:
                label_list.append(0)
            data_list.append([int(int(feature)>128) for feature in curline[1:]])

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list)
    return data_matrix, label_matrix
    
if __name__ == '__main__':
    start = time.time()

    print('start read transSet')
    trainData, trainLabel = loadData('D:/mnist_train/mnist_train.csv')

    print('start read testSet')
    testData, testLabel = loadData('D:/mnist_train/mnist_test.csv')

    max_entropy = max_entropy(trainData[:20000], trainLabel[:20000], testData, testLabel)

    max_entropy.maxEntropyTrain()

    accuracy = max_entropy.test()
    print(f"the accuracy is {accuracy}.")
    print('time span:', time.time() - start)