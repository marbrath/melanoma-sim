import numpy as np
import csv
import itertools

fam_events = np.load('npy_files/fam_events.npy')

num_children = 8
arr = np.zeros([2]*(2 + num_children))

for i in range(0, len(fam_events)):
	idx = fam_events[i]
	arr[tuple(idx)] += 1

perm = list(itertools.product([0,1], repeat=10))



for i in range(0,9):
	print('Num sick: ', i)

	for p in perm:
		num_fam = arr[p]

		if (sum(p)==i and num_fam > 0):
			print("Constellation: ", p, "num: ", num_fam)
	
##test
'''
combination = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(combination)
counter = 0

for i in range(0, len(fam_events)):
	if (np.array_equal(fam_events[i], combination)):
		counter += 1

print(counter)
'''

'''

num_daughters = np.load('npy_files/num_daughters.npy')
num_sons = np.load('npy_files/num_sons.npy')
num_parents = np.load('npy_files/num_parents.npy')
num_children = len(num_daughters)

n = max(num_daughters + num_sons)
table = np.zeros((n + 1, n + 1, 3), dtype =np.int64)

for l in range(0, num_children):
	i = num_daughters[l]
	j = num_sons[l]
	k = num_parents[l] - 1

	table[i][j][k] += 1


with open('family_sizes.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["no children", "no daughters", "no sons", "both parents", "father only", "mother only", "total"])
	for l in range(0, n+1):
		for i in range(0, n+1):
			for j in range(0, n+1):
				if (i + j == l):
					num_families = table[i][j][0] + table[i][j][1] + table[i][j][2]
					if (num_families > 0):
							writer.writerow([l, i, j, table[i][j][2], table[i][j][1], table[i][j][0], num_families])

'''

