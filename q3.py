import numpy as np
x=np.array([[2,3],[3,4]])
print("cube using multiple fn",np.multiply(x,np.multiply(x,x)))
print("cube using power fn",np.power(x,3))
print("cube using **",x**3)
print("cuube using 8",x*x*x)
print('identity matrix ',np.identity(2,dtype=int))
print("display each element of the matrix to different powers",np.power(x,[[1,2],[3,4]]))
y=np.array([[2,6],[3,9]])
print((x**2)+(2*y))