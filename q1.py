import numpy as np
m1=np.array([[1,2,3],[6,7,8],[1,4,5]])
m2=np.array([[2,6,4],[8,5,4],[4,6,2]])
print("matrix m1",m1)
print("matrix m2",m2)
r1=np.add(m1,m2)
print("sum of matrix")
print(r1)
r5=np.matmul(m1,m2)
print("multiplication",r5)
print("transpose of m1",np.transpose(m1))
      
