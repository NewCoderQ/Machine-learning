# -*- coding: utf-8 -*-

from numpy import *
import re

def loadDataSet():
	postingList = []
	f = open('src_file.txt', 'r')
	line = f.readline()
	while line:
		line_list = line.strip().split(' ')
		postingList.append(line_list)
		line = f.readline()

	f.close()
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec


def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setofWords2Vec(vocabList, inputSet):					# set of words model
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print 'the word: %s is not in my Vocabulary' % word

	return returnVec

def bagOfWords2VecMN(vocabList, inputSet):					# bag of words model
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print 'the word: %s is not in my Vocabulary' % word

	return returnVec

def trainNBO(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])									# 
	pAbusive = sum(trainCategory) / float(numTrainDocs)				# 0.5
	# pONum = zeros(numWords); p1Num = zeros(numWords)				# numerator
	# pODenom = 0.0; p1Denom = 0.0									# denominator 
	pONum = ones(numWords); p1Num = ones(numWords)				# numerator
	pODenom = 2.0; p1Denom = 2.0									# denominator 
	for i in range(numTrainDocs):									
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			pONum += trainMatrix[i]
			pODenom += sum(trainMatrix[i])

	# p1Vect = p1Num / p1Denom
	# pOVect = pONum / pODenom

	p1Vect = log(p1Num / p1Denom)
	pOVect = log(pONum / pODenom)

	return pOVect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

# **************************process file**********************************
def testParse(bigString):
	listOfTokens = re.split(r'\w*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1, 26):
		wordList = testParse(open(r'email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)							# add element, return a list
		classList.append(1)

		wordList = testParse(open(r'email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)							# add element, return a list
		classList.append(0)

	vocabList = createVocabList(docList)
	trainingSet = range(50); testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setofWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWOrds2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1

	print 'the error rate is : ', float(errorCount) / len(testSet)
	

if __name__ == '__main__':
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setofWords2Vec(myVocabList, postinDoc))

	p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setofWords2Vec(myVocabList, testEntry))
	print thisDoc
	print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)
