# -*- coding: utf-8 -*- 

'''
	Machine Learning in Active
	Logistic Regressiion
	Date: 06/04/2017
'''

import numpy as np
import matplotlib.pyplot as plt

# load dataSet
def loadDataSet():
	dataMat = []; labelMat = []					# init dataSet and labelSet
	fr = open('testSet.txt')					# open txt file

	for line in fr.readlines():					# read txt file
		lineArr = line.strip().split()			# split line into list
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		# add data to dataMat
		labelMat.append(int(lineArr[2]))		# add label to labelMat

	return dataMat, labelMat

# define Sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

# gradient ascent algorithm
def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()	# transpose the label matrix
	m, n = np.shape(dataMatrix)					# get the number of row and column of the dataMatrix, 100 * 3
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n, 1))					# create a n * 1 matrix, 3 * 1
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights


# draw the best fit
def drawBestFit(weights):
	dataMat, labelMat = loadDataSet()			# load data set
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]					# get the number of row of the dataArr
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = np.arange(-3.0, 3.0, 0.1)				# the split line
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.show()


# stochastic gradient ascent
def stocGradAscent0(dataMatrix, classLabels):
	m,n = np.shape(dataMatrix)
	alpha = 0.01
	weights = np.ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights


# improved stochastic gradient ascent
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m, n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(np.random.uniform(0, len(dataIndex)))		# create random index
			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])

	return weights

# ***********************************horse prediction*************************************
def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5: 
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')

	# train 
	trainingSet = []; trainingLabel = []
	for line in frTrain.readlines():
		currentLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):							# the first 21 values
			lineArr.append(float(currentLine[i]))	# save into a list
		trainingSet.append(lineArr)
		trainingLabel.append(float(currentLine[21]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabel, 500)
	
	# test
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currentLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currentLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights)) != int(currentLine[21]):
			errorCount += 1

	# result
	errorRate = (float(errorCount) / numTestVec)
	print 'the error rate of this test is: %f' % errorRate
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print 'after %d iterations the average error rate is %f' % (numTests, errorSum / float(numTests))


if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	# drawBestFit(gradAscent(dataArr, labelMat).getA())
	# drawBestFit(stocGradAscent0(np.array(dataArr), labelMat))
	# drawBestFit(stocGradAscent1(np.array(dataArr), labelMat))
	multiTest()
