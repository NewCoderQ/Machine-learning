# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
import loadData


'''
	随机选择alpha

	parameters:
		i: 第一个alpha的索引值
		m: alpha参数的个数

	return:
		j: 第二个alpha的索引值

'''
def selectJrand(i, m):
	j = i
	while (j == i):			# 使得选取的两个参数的索引不相同
		j = int(random.uniform(0, m))
	return j


def clipAlpha(aj, H, L):
	'''
		修剪alpha的值，使得计算生成的alpha在允许的范围内L - H

		parameters:
			aj: alpha_j的值
			H：	alpha上限
			L： alpha下限
	'''
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	'''
		简化版SMO算法

		parameters:
			dataMatIn: 数据矩阵
			classLabels: 标签矩阵
			C: 松弛变量
			toler: 容错率
			maxIter: 最大迭代次数

		returns:
			b: 偏置量
			alphas: 更新完成的alpha值的列表	
	'''

	# 添加动态直线图
	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMatIn)):
		if classLabels[i] > 0:
			data_plus.append(dataMatIn[i])
		else:
			data_minus.append(dataMatIn[i])
	data_plus_np = np.array(data_plus)              #转换为numpy矩阵
	data_minus_np = np.array(data_minus)            #转换为numpy矩阵


	# 将数据和标签转换成np.mat的形式
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()	# 此处标签需要进行转置
	b = 0; m, n = np.shape(labelMat)		# 初始化b，并且获取数据的维度
	# 初始化alpha参数，初始设为0, alpha参数的个数和数据点的个数相同，也就是跟数据的行数相同
	alphas = np.mat(np.zeros((m, 1)))		# alphas: (m, 1)
	# 循环计算
	iter_num = 0	

	while(iter_num < maxIter):
		plt.clf()
		alphaPairsChanged = 0		# 创建一个新的变量，用来纪录alpha对变化的次数
		for i in range(m):	
			# 首先计算f(xi)
			fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
			Ei = fxi - float(labelMat[i])	# 计算误差

			# 优化alpha，设置一定的容错率
			# 挑选出那些 不符合要求的alpha值，对其进行计算更新
			if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
				# 随机选择一个alp	ha_j组成alpha对来进行更新
				j = selectJrand(i, m)
				# 计算fxj
				fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
				# 误差
				Ej = fxj - float(labelMat[j])
				# 保存更新前的alpha的值
				alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()

				# 根据表签来计算alpha的上界和下界
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:	# 标签相同
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[j])
				# 对L == H的情况做特殊处理
				if (L == H): print('L == H'); continue

				# 计算eta, 即学习率
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
						- dataMatrix[j, :] * dataMatrix[j, :].T

				# 判断eta是否大于0
				if eta >= 0: print('eta >= 0'); continue

				# 更新alpha_j
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				# clip the alpha_j
				alphas[j] = clipAlpha(alphas[j], H, L)
				# 判断alpha的值是否变化太小
				if (abs(alphas[j] - alphaJold) < 0.00001): print('alpha_j的变换量太小'); continue

				# 更新alpha_i
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

				# 更新b1和b2
				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
						- labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
				b2 = b - Ej- labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T \
						- labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T

				# 根据b1 b2的值更新b
				if ((0 < alphas[i]) and (C > alphas[i])): b = b1
				elif ((0 < alphas[j]) and (C < alphas[j])): b = b2
				else: b = (b1 + b2) / 2.0

				# 统计优化的次数
				alphaPairsChanged += 1
				# print
				print("第%d次迭代样本： %d, alpha优化次数：%d" % (iter_num, i, alphaPairsChanged))

		# 更新迭代次数
		if (alphaPairsChanged == 0): iter_num += 1
		else: iter_num = 0
		print("迭代次数：%d" % iter_num)

		w = get_w(dataMatIn, classLabels, alphas)		# 计算w的值

		plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c = 'red', s=30, alpha=0.7)   #正样本散点图
		plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c = 'black', s=30, alpha=0.7) #负样本散点图
		x1 = max(dataMatIn)[0]	# 找到dataMat中x的最大值
		x2 = min(dataMatIn)[0]	# 找到dataMat中x的最小值
		a1, a2 = w 				# 将w中的两个值分别赋值给a1, a2
		# 将数据转换成float类型
		b = float(b)
		a1 = float(a1[0])		
		a2 = float(a2[0])
		y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
		plt.plot([x1, x2], [y1, y2])
		#找出支持向量点
		for i, alpha in enumerate(alphas):
			if abs(alpha) > 0:
				x, y = dataMat[i]
				plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
		plt.xlim((-2, 12))
		plt.ylim((-10, 10))
		plt.pause(0.0000001)
		# plt.close()
		# w = get_w(dataMat, labelMat, alphas)
		# print(w)
		# showClassifer(dataMat, w, b, alphas)

	return b, alphas


def showClassifer(dataMat, w, b, alphas):
	#绘制样本点
	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)              #转换为numpy矩阵
	data_minus_np = np.array(data_minus)            #转换为numpy矩阵

	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c = 'red', s=30, alpha=0.7)   #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c = 'black', s=30, alpha=0.7) #负样本散点图
	#绘制直线
	'''
		根据间隔最大的横坐标值计算出直线对应的纵坐标的值
		即可计算出两个点的坐标
		然后根据这两个点的坐标画出对应的直线
	'''
	x1 = max(dataMat)[0]	# 找到dataMat中x的最大值
	x2 = min(dataMat)[0]	# 找到dataMat中x的最小值
	a1, a2 = w 				# 将w中的两个值分别赋值给a1, a2
	# 将数据转换成float类型
	b = float(b)
	a1 = float(a1[0])		
	a2 = float(a2[0])
	y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
	plt.plot([x1, x2], [y1, y2])
	#找出支持向量点
	for i, alpha in enumerate(alphas):
		if abs(alpha) > 0:
			x, y = dataMat[i]
			plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
	plt.show()



# 计算w的值，次数是一个2 * 1的矩阵
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
	dataMat, labelMat = loadData.loadDataSet('testSet.txt')
	b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
	w = get_w(dataMat, labelMat, alphas)
	showClassifer(dataMat, w, b, alphas)
	

