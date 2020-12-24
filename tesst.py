import numpy as np

a = [np.array([[2, 3],[1, 1]]), np.array([[2, 2],[2, 2]]), np.array([[3, 3],[6, 3]])]

b = np.zeros((a[0].shape[0], a[0].shape[1]))
for i in range(0,len(a)):
    b +=a[i]

l = np.where(b == np.max(a))[0][0]
print(l)