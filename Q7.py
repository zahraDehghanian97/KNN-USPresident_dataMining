import csv
import random
import math
import operator
import matplotlib.pyplot as plt


def handleDataset(filename, trainingSet=[], testSet=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1, len(dataset) - 1):
            rand = random.randint(1, 3)
            if rand > 2:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1, length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] is predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def knn(k):
    # prepare data
    print('start KNN algorithm with K = ' + str(k))
    trainingSet = []
    testSet = []
    handleDataset("US Presidential Data.csv", trainingSet, testSet)
    print('size Train set: ' + repr(len(trainingSet)))
    print('size Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    return accuracy

def a():
    accuracy=[]
    for i in range(10):
        accuracy.append(knn(1))
    print("run K-NN with K=1 10 times :")
    min = 0
    for i in range(10):
        print(">> round "+str(i)+" : accuracy = "+str(accuracy[i]))
        min = min + accuracy[i]
    min = min /10
    print('mean value = '+str(min))
def b():
    print('>> Question Number 7')
    print('>>evaluate KNN for k in 5, 10, 15, 20, 25 ')
    Accuraccy = []
    x=[]
    for j in range(3)  :
        x=[]
        for i in (1,5,10, 15, 20, 25):
            x.append(knn(i))
        Accuraccy.append(x)
    x=[]
    print(Accuraccy)
    for j in range(6):
        mid =0
        for i in range(3):
            mid=mid+ Accuraccy[i][j]
        mid=mid/3
        x.append(mid)
    print(x)
    plt.plot([1,5, 10, 15, 20, 25], x)
    plt.ylabel('Accuracy')
    plt.show()


a()
b()
