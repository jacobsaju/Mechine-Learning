import numpy as np
array=np.random.randint(10,size=(3,3))
print("square matrix",array)
print("inverse matrix",np.linalg.inv(array))
rank=np.linalg.matrix_rank(array)
print("Rank",rank)
