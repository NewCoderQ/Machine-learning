# -*- coding: utf-8 -*-

'''
	Machine Learning in Action
	Support Vector Machines
	Date: 07/04/2017
'''
import numpy as np

# load data set
def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat


# create random int value
def selectJrand(i, m):
	j = i
	while (j == i):								# j != i
		j = int(np.random.uniform(0, m))
	return j

# adjust alpha
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj  

# sequential minimal optimization
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = np.mat(dataMatIn)					# 100 * 2
	labelMat = np.mat(classLabels).transpose()		# 100 * 1
	b = 0; m, n = np.shape(dataMatrix)				# 100, 2
	alphas = np.mat(np.zeros((m, 1)))				# 100 * 1, 0
	iter = 0
	index = 0
	while (iter < maxIter):							# condition
		index += 1
		alphaPairsChanged = 0
		for i in range(m):
			fXi = float(np.multiply(alphas, labelMat).T * 					# m * m transpose	
					   	(dataMatrix * dataMatrix[i, :].T)) + b				# predict label
			# print np.shape(dataMatrix[i, :])	
			Ei = fXi - float(labelMat[i])									# error value
			if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
				((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
				j = selectJrand(i, m)										# select another alpha
				fXj = float(np.multiply(alphas, labelMat).T *				# predict label
							(dataMatrix * dataMatrix[j, :].T)) + b
				Ej = fXj - float(labelMat[j])								# error value
				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()
				if (labelMat[i] != labelMat[j]):							# two labels are different
					L = max(0, alphas[j] - alphas[i])						# max
					H = min(C, C + alphas[j] - alphas[i])					# min
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])

				print "index = ", index, "L = ", L, "H = ", H
				if L == H: print "L == H", L, alphas[j], alphas[i]; continue
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
						dataMatrix[i, :] * dataMatrix[i, :].T - \
						dataMatrix[j, :] * dataMatrix[j, :].T
				if eta >= 0: print "eta >= 0"; continue
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				alphas[j] = clipAlpha(alphas[i], H, L)
				if(abs(alphas[j] - alphaJold) < 0.00001):
					print "j not moving enouth"; continue
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
					dataMatrix[i, :] * dataMatrix[i, :].T - \
					labelMat[j] * (alphas[j] - alphaJold) * \
					dataMatrix[i, :] * dataMatrix[j, :].T
				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
					dataMatrix[i, :] * dataMatrix[j, :].T - \
					labelMat[j] * (alphas[j] - alphaJold) * \
					dataMatrix[j, :] * dataMatrix[j, :].T
				if (0 < alphas[i]) and (C > alphas[i]) : b = b1
				elif (0 < alphas[j]) and (C > alphas[j]) : b = b2
				else: b = (b1 + b2) / 2.0
				alphaPairsChanged += 1
				print "iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
		if (alphaPairsChanged == 0):
			iter += 1
		else:
			iter = 0
		print "iteration number: %d" % iter
	return b, alphas


# main
if __name__ == '__main__':
	dataArr, labelArr = loadDataSet('testSet.txt')
	b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	print b
