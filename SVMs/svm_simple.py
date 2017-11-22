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

'''
	修剪alpha的值，使得计算生成的alpha在允许的范围内L - H

	parameters:
		aj: alpha_j的值
		H：	alpha上限
		L： alpha下限
'''
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj

'''
	简化版SMO算法

	parameters:
		dataMatIn: 数据矩阵
		classLabels: 标签矩阵
		C: 松弛变量
		toler: 容错率
		maxIter: 最大迭代次数
'''





