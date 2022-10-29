import numpy as np

a = np.array([[0, 0, 1, -1], [0, 3, 3, 4], [0, -2, 1, 7]])
b = np.array([-2, -3, 0, 4])

result = a.dot(b)
print(result)
