import numpy as np

def matrices_multi(A, B):

    if not A or not A[0] or not B or not B[0]:
        return 0

    Arow,Brow,Acol,Bcol = len(A),len(B),len(A[0]),len(B[0])

    if Arow != Bcol or Acol != Brow:
        raise ValueError
    
    
    C = [[0 for m in range(Bcol)] for n in range(Arow)]
    for i in range(Arow):
        for j in range(Bcol):
            for k in range(Acol):
                C[i][j] += A[i][k] * B[k][j]

    return C



"""
#test
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
C = matrices_multi(A, B)
Arow,Brow,Acol,Bcol = len(A),len(B),len(A[0]),len(B[0])
print(len(A),len(B),len(A[0]),len(B[0]))
print(C)
print(np.array(C))

A = []
B = []
C = matrices_multi(A, B)
A_np = np.array(A)
B_np = np.array(B)
C = matrices_multi(A, B)
print(C)
print(np.array(C))
print(np.dot(A,B))
print(np.array(C) == np.dot(A,B))


A = np.random.rand(5,5).tolist()
B = np.random.rand(5,5).tolist()
C = matrices_multi(A, B)
A_np = np.array(A)
B_np = np.array(B)
print(C)
print(np.array(C))
print(np.dot(A,B))
print(np.array(C) == np.dot(A,B))
"""