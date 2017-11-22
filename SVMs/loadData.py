# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

"""
	该模块提供数据读取函数
	可以从原始数据读入数据整理成分类器所需要的格式
"""

# 读取数据
def loadDataSet(filename):
	# 为数据集创建两个列表，用来存放数据和标签
	dataMat = []; labelMat = []	
	fr = open(filename)		# 打开文件
	for line in fr.readlines():		# 遍历文件中的每一行
		lineArr = line.strip().split('\t')	# 将每一行的元素进行切分，前两个元素为data最后一个元素为标签
		dataMat.append([float(lineArr[0]), float(lineArr[1])])	# data
		labelMat.append(float(lineArr[2]))	# label
	return dataMat, labelMat

# 数据可视化
def showDataSet(dataMat, labelMat):
	data_plus = []		# 正样本
	data_minus = []		# 负样本

	# 遍历数据集中的每一个数据，根据标签获取他们的样本类别
	# 将正负样本分别存进不同的列表中
	for i in range(len(dataMat)):
		if labelMat[i] > 0:	# 正样本
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])

	# 将数据集从列表转换成array类型
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)

	print(np.array(data_minus_np))
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c = 'red')
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c = 'black')
	plt.show()

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('testSet.txt')
	showDataSet(dataMat, labelMat)






