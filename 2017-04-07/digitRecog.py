# -*- coding: utf-8 -*-

import svmMLiA
import numpy as np

# loadImage
def loadImages(dirName):
	from os import listdir						# import os
	hwLabels = []								# handwriting label
	trainingFileList = listdir(dirName)			# training file names list
	print trainingFileList
	m = len(trainingFileList)					# get the number of the training files
	trainingMat = np.zeros((m, 1024))			# m * 1024
	for i in range(m):							# each training file
		fileNameStr = trainingFileList[i]		# file name str(including extension)
		fileStr = fileNameStr.split('.')[0]		# file name str
		classNumStr = int(fileStr.split('_')[0])		# get the number
		if classNumStr == 9: hwLabels.append(-1)		# 只做一个二分类，分为-1 和 +1
														# 9标签为-1，其他数的标签都为1
		else: hwLabels.append(1)
		trainingMat[i, :] = img2vector("%s/%s" % (dirName, fileNameStr))	
	return trainingMat, hwLabels


# image to vector
def img2vector(filename):
	returnVect = np.zeros((1, 1024))			# 1 * 1024
	fr = open(filename)							# open file
	for i in range(32):							# each line
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32 * i +j] = int(lineStr[j])		# 将每一个的每一个元素都转换成int值存储在returnVect中
	return returnVect

# train digits
def trainDigits(kTup = ('rbf', 10)):
	dataArr, labelArr = loadImages('trainingDigits')
	b, alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)		# calculate b and alpha
	dataMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A > 0)[0]
	sVs = dataMat[svInd]									# get support vector
	labelSV = labelMat[svInd]								# get labels of support vector
	print "there are %d Support vectors" % np.shape(sVs)[0]
	m,n = np.shape(dataMat)
	errorCount = 0
	for i in range(m):
		kernelEval = svmMLiA.kernelTrans(sVs, dataMat[i, :], kTup)
		predict = kernelEval.T * np.multiply(alphas[svInd], labelSV) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print "the training error rate is: %f" % (float(errorCount) / m)
	
	# test error
	dataArr, labelArr = loadImages('testDigits')
	errorCount = 0
	dataMat = np.mat(dataArr); labelMat = np.mat(labelArr)
	m, n = np.shape(dataMat)
	for i in range(m):
		kernelEval = svmMLiA.kernelTrans(sVs, dataMat[i, :], kTup)
		predict = kernelEval.T * np.multiply(alphas[svInd], labelSV) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print "the test error rate is: %f" % (float(errorCount) / m)


if __name__ == '__main__':
	trainDigits(('rbf', 10))


