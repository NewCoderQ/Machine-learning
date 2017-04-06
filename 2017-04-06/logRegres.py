# -*- coding: utf-8 -*- 

'''
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


if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	drawBestFit(gradAscent(dataArr, labelMat).getA())
