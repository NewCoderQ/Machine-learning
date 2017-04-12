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

	if aj < L:
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
		if (alphaPairsChanged == 0):
			iter += 1
		else:
			iter = 0
		print "iteration number: %d" % iter
	return b, alphas

# 完整版的Platt SMO的支持函数
# 创建一个数据结构，用来保存计算需要的信息
class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn								# 存放输入的数据矩阵
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = np.shape(dataMatIn)[0]					# 获取输入数据矩阵的行数
		self.alphas = np.mat(np.zeros((self.m, 1)))		# 创建alphas, 全零矩阵
		self.b = 0										# 将计算改变量初始化为0
		self.eCache = np.mat(np.zeros((self.m, 2)))		# 创建一个误差缓存变量，用于比较误差值的大小

# 计算误差
def calcEk(oS, k):
	fXk = float(np.multiply(oS.alphas, oS.labelMat).T *
				(oS.X * oS.X[k, :].T)) + oS.b
	Ek = fXk - float(oS.labelMat[k])
	return Ek

# 选择第二个alpha的值
def selectJ(i, oS, Ei):
	maxK = -1; maxDeltaE = 0; Ej = 0					# 初始化
	oS.eCache[i] = [1, Ei]								# 改变误差缓存中指定位置的值
	validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]	# 返回误差缓存中第一列元素不为0的位置信息，取行数
	if (len(validEcacheList)) > 1:						# 误差缓存中至少有两个误差不为0
		for k in validEcacheList:
			if k == i: continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			if (deltaE > maxDeltaE):					# 选择误差改变值最大的
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else:
		j = selectJrand(i, oS.m)						# 随机生成一个j
		Ej = calcEk(oS, j)
	return j, Ej

# 更新误差缓存，将计算好的误差存入缓存中
def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]


# 完整的Platt SMO算法中的优化例程
def innerL(i, oS):
	Ei = calcEk(oS, i)									# 计算误差
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
		((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):			# 真实标签与预测标签之间的误差不在容差率之内
																			# 并且alpha值在(0, C)之间，C为一个具体的常数
		j, Ej = selectJ(i, oS, Ei)											# 随机选择任一个其他的alpha值，select another alpha
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])							# L的最小值为0
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])				# H的最大值为oS.C
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)					# L的最小值为0
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])						# H的最大值为oS.C

		if L == H: print "L == H"; return 0									# 如果L == H，退出
		eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
				oS.X[j, :] * oS.X[j, :].T 									# 最优修改量
		if eta >= 0: print "eta >= 0"; return 0

		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)						# 将oS.alphas[j]控制在 L ~ H 之间
		updateEk(oS, j)								# 重新计算误差，将计算好的误差存入到误差缓存中
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):						# J的变化幅度不满足要求
			print "J not moving enough"; return 0
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
						(alphaJold - oS.alphas[j])							# 计算alphas[i]
		updateEk(oS, i)														# 更新误差缓存
		

		# 计算新的偏移量
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
				oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
				(oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
				oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
				(oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2) / 2.0
		return 1
	else: return 0

# 完整版Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)			# 创建实例
	iter = 0
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
		alphaPairsChanged = 0
		if entireSet:
			for i in range(oS.m):									# 输入数据的行数
				alphaPairsChanged += innerL(i, oS)
			print "fullSet, iter: %d i: %d, pairs changed %d" % \
					(iter, i, alphaPairsChanged)
			iter += 1
		else:
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
				print "non-bound, iter: %d i: %d, pairs changed %d" % \
						(iter, i, alphaPairsChanged)
			iter += 1

		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet = True
		print "iteration number: %d" % iter
	return oS.b, oS.alphas



# main
if __name__ == '__main__':
	dataArr, labelArr = loadDataSet('testSet.txt')
	# b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	# print b, alphas[alphas > 0]
	# for i in range(100):
		# if alphas[i] > 0.0 : print dataArr[i], labelArr[i]

	b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
