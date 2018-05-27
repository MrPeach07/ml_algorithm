#################################################  
# kmeans: k-means cluster  
# Author : MrPeach07  
# Date   : 2018-5-28  
# HomePage : https://blog.csdn.net/MR_Peach07 
# Email  : lizhiyu_9709@163.com  
################################################# 

import numpy as np
import random
import re
import matplotlib.pyplot as plt  

def show_fig():
    dataSet = loadDataSet()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1])
    plt.show()

# 计算两个向量间的欧式距离
def calcuDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def loadDataSet():
    dataSet = np.loadtxt('data/dataSet.txt')
    return dataSet

# 从数据集中随机选取k个数据返回
def initCentroids(dataSet, k):
    dataSet = list(dataSet)
    return random.sample(dataSet, k)

# 对每个属于dataSet的项,计算每个项与质心列表中k个质心的距离，找出距离最小的，并将该项加入到相应的cluster中
def minDistance(dataSet, centroidList):
    #dict保存clsuter结果
    clusterDict = dict()
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        flag = -1
        minDis = float("inf")  # 初始化为最大值
        for i in  range(k):
            vec2 = centroidList[i]
            distance = calcuDistance(vec1, vec2)
            if distance < minDis:
                minDis = distance
                flag = i  # 循环结束时，flag保存与当前项最近的cluster标记
        if flag not in  clusterDict.keys():
            clusterDict.setdefault(flag, [])
        clusterDict[flag].append(item) # 加入相应的类别中
    return clusterDict # 不同的类别

# 重新计算各个质心
def getCentroids(clusterDict):
    centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList # 得到新的质心

# 计算各簇集合间的均方误差
# 将簇类中各个向量与质心的距离累加求和
def getVar(centroidList, clusterDict):
    sum = 0.0
    for key in clusterDict.keys():
        vec1 = centroidList[key]
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = item
            distance += calcuDistance(vec1, vec2)
        sum += distance
    return sum

# 利用plt展示聚类结果
def showCluster(centroidList, clusterDict):
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow'] # 不同cluster标记，o表示圆形，另一个表示颜色。
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], colorMark[key], markersize=12)  # 质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])
    plt.show()

def test_k_means():
    dataSet = loadDataSet()
    centroidList = initCentroids(dataSet, 4)
    clusterDict = minDistance(dataSet, centroidList)
    # # getCentroids(clusterDict)
    # showCluster(centroidList, clusterDict)
    newVar = getVar(centroidList, clusterDict)
    oldVar = 1 # 当两次聚类的误差小于某个值时，说明质心基本确定。

    tiems = 2
    while abs(newVar - oldVar) >= 0.00001:
        centroidList = getCentroids(clusterDict)
        clusterDict = minDistance(dataSet, centroidList)
        oldVar = newVar
        newVar = getVar(centroidList, clusterDict)
        times = 1
        showCluster(centroidList, clusterDict)

if __name__ == '__main__':
    # show_fig()
    test_k_means()