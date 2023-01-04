import numpy as np
from numpy.linalg import inv
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.array([-2,5,-8])
y=np.linalg.inv(A)
x=np.linalg.solve(y,b)
print(format(x))