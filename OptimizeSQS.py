import numpy as np
import random

numberOfTrials = 100
numberOfS = 5
threshold = 0.01


A = []
with open('A_test.txt','r') as inFile:
	for line in inFile:
		data = line.replace('[','')
		data = data.replace(']','')
		data = data.replace('.','')
		tmp = []
		for d in data.split():
			tmp.append(int(d))
		A.append(tmp)

def createRandomS():
	S = []
	for i in range(len(tmp)):
		S.append(1)


	while len([a for a in S if a == -1]) < len(S)/2:
		rn = random.randint(1,len(tmp))
		S[rn-1] = -1

	return S

#print S

def calculateNorm(S,A):
	S = np.array(S)
	St = np.transpose(S)
	A = np.array(A)
	r = np.dot(np.dot(St,A),S)

	return abs(r)


def switchAtoms(S):
	plusOnes, minusOnes = [],[]
	for i in range(len(S)):
		if S[i] == 1:
			plusOnes.append(i)
		else:
			minusOnes.append(i)
	rn1 = random.randint(0,len(plusOnes))
	rn2 = random.randint(0,len(minusOnes))

	S[rn1] = -1
	S[rn2] = 1

	return S


saved_S = []


while len(saved_S) < numberOfS:
	old_sqs = 1000
	S = createRandomS()
	n = 0
	print('one more run')
	while n < numberOfTrials:
		new_S = switchAtoms(S) # Here, we randomly swith a +1 and a -1
		new_sqs = calculateNorm(S,A)

		if new_sqs < old_sqs:
			accept = True
		else:
			rn = random.random()
			if rn > 0.5:
				accept = True
		if accept:
			S = new_S

		old_sqs = new_sqs

		# Everytime we find a good S, we keep it and start over by reinitializing old_sqs to a large number
		if new_sqs <= threshold:
			if S not in saved_S:
				saved_S.append(S)
				print(S)
			break

		n += 1
