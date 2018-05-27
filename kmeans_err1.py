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

# ���������������ŷʽ����
def calcuDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def loadDataSet():
    dataSet = np.loadtxt('data/dataSet.txt')
    return dataSet

# �����ݼ������ѡȡk�����ݷ���
def initCentroids(dataSet, k):
    dataSet = list(dataSet)
    return random.sample(dataSet, k)

# ��ÿ������dataSet����,����ÿ�����������б���k�����ĵľ��룬�ҳ�������С�ģ�����������뵽��Ӧ��cluster��
def minDistance(dataSet, centroidList):
    #dict����clsuter���
    clusterDict = dict()
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        flag = -1
        minDis = float("inf")  # ��ʼ��Ϊ���ֵ
        for i in  range(k):
            vec2 = centroidList[i]
            distance = calcuDistance(vec1, vec2)
            if distance < minDis:
                minDis = distance
                flag = i  # ѭ������ʱ��flag�����뵱ǰ�������cluster���
        if flag not in  clusterDict.keys():
            clusterDict.setdefault(flag, [])
        clusterDict[flag].append(item) # ������Ӧ�������
    return clusterDict # ��ͬ�����

# ���¼����������
def getCentroids(clusterDict):
    centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList # �õ��µ�����

# ������ؼ��ϼ�ľ������
# �������и������������ĵľ����ۼ����
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

# ����pltչʾ������
def showCluster(centroidList, clusterDict):
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow'] # ��ͬcluster��ǣ�o��ʾԲ�Σ���һ����ʾ��ɫ��
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], colorMark[key], markersize=12)  # ���ĵ�
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
    oldVar = 1 # �����ξ�������С��ĳ��ֵʱ��˵�����Ļ���ȷ����

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