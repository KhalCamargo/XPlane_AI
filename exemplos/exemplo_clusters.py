import os 
import ipdb; 
import numpy as np 
import matplotlib.pyplot as plt
import skfuzzy as fuzz 

X = np.array([[2, 2], [2, 3], [1, 4], [7, 2], [6, 6], [6, 4]])

#plt.scatter(X[:,0],X[:,1]) 
#plt.show() 
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, c=2, m=2, error=0.005, maxiter=1000, init=None) 

# print the results for the training data 
print(u) 
# predict for new data 
# kmeanss.predict([[0, 0]]) 
# kmeans.predict([[0,0], [12,3]]) 

newdata = np.array([[3, 4], [5, 6], [3, 5]]) 

u_new, u0_new, d_new, jm_new, p_new, fpc_new = fuzz.cluster.cmeans_predict(newdata.T, cntr, 2, error=0.005, maxiter=1000) 

output_clusters = np.argmax(u_new, axis=0) 

print(output_clusters)