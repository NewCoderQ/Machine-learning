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
# 参数：数据集，类别标签，常数C，容错率，推出前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = np.mat(dataMatIn)					# 100 * 2
	labelMat = np.mat(classLabels).transpose()		# 100 * 1			转置label矩阵
	b = 0; m, n = np.shape(dataMatrix)				# 100, 2
	alphas = np.mat(np.zeros((m, 1)))				# 100 * 1的全零矩阵
	iter = 0										# 将循环次数初始化为0
	index = 0	
	while (iter < maxIter):							# condition
		index += 1
		alphaPairsChanged = 0						# alpha对改变的次，用于记录alpha是否进行了优化
		for i in range(m):							# [0, 100)
			fXi = float(np.multiply(alphas, labelMat).T * 					# m * m transpose	
					   	(dataMatrix * dataMatrix[i, :].T)) + b				# predict label, 预测标签值
			# # print np.shape(dataMatrix[i, :])	
			Ei = fXi - float(labelMat[i])									# error value 计算预测标签与真实标签之间的误差值
			if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
				((labelMat[i] * Ei > toler) and (alphas[i] > 0)):			# 真实标签与预测标签之间的误差不在容差率之内
																			# 并且alpha值在(0, C)之间，C为一个具体的常数
				j = selectJrand(i, m)										# 随机选择任一个其他的alpha值，select another alpha
				fXj = float(np.multiply(alphas, labelMat).T *				# predict label 预测标签
							(dataMatrix * dataMatrix[j, :].T)) + b
				Ej = fXj - float(labelMat[j])								# error value	求误差
				alphaIold = alphas[i].copy()								# 0. 复制一份alpha改变之前的值
				alphaJold = alphas[j].copy()

				# 保证alpha的值在0和C之间
				if (labelMat[i] != labelMat[j]):							# two labels are different
					L = max(0, alphas[j] - alphas[i])						# L的最小值为0
					H = min(C, C + alphas[j] - alphas[i])					# H的最大值为C
				else:
					L = max(0, alphas[j] + alphas[i] - C)					# L的最小值为0
					H = min(C, alphas[j] + alphas[i])						# H的最大值为C

				# # print "index = ", index, "L = ", L, "H = ", H
				if L == H: 
					print "L == H", L, alphas[j], alphas[i]
					continue

				# 计算alpha的最优修改量
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
						dataMatrix[i, :] * dataMatrix[i, :].T - \
						dataMatrix[j, :] * dataMatrix[j, :].T
				# print "eta = ", eta
				if eta >= 0: print "eta >= 0"; continue
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta				# 计算 alpha
				alphas[j] = clipAlpha(alphas[j], H, L)					# 将alpha值与H和L进行比较，如果在范围之外则重新进行赋值
				
				# 检查alpha[j]是否有轻微的变化，如果没有的话
				if(abs(alphas[j] - alphaJold) < 0.00001):				# 检查更新后的alpha与初始的alpha值的差值
					print "j not moving enouth"; 
					continue

				# 对alpha[i]进行改变，改变的方向与alpha[j]的方向相反
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

				# 对新生成的alpha[i],alpha[j]设置新的b值，进行下一次计算
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
				alphaPairsChanged += 1						# alpha的值改变了就 +1
				print "iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
		if (alphaPairsChanged == 0):						# 继续循环
			iter += 1
		else:												# 重新进行循环
			iter = 0
		print "iteration number: %d" % iter
	return b, alphas


# main
if __name__ == '__main__':
	dataArr, labelArr = loadDataSet('testSet.txt')
	b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	print b, alphas[alphas > 0]
