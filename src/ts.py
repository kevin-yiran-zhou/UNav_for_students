import numpy as np

a=np.sqrt(2)/2
b=np.sqrt(3)/2

A=np.array([[1,0,0],[0,a,-a],[0,a,a]])
B=np.array([[1/2,0,-b],[0,1,0],[b,0,1/2]])
C=np.array([[b,1/2,0],[-1/2,b,0],[0,0,1]])
D=np.array([[1,0,0],[0,a,a],[0,-a,a]])

E=D@C@B@A

print(E)