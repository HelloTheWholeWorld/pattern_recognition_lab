from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

nGauss = 1

X = np.loadtxt(open("TrainSamples-17.csv","rb"),delimiter=",",skiprows=0)
Y = np.loadtxt(open("TrainLabels-17.csv","rb"),delimiter=",",skiprows=0)

z = 0;
GMMs = []
for i in [0,1,2,3,4,5,6,7,8,9]:
    id = [j for j in range(len(Y)) if Y[j] == i]
    XX = X[id,:]
    YY = Y[id]
    print( "Training GMM%d...\n" %i)
    gmm = GaussianMixture(n_components=nGauss, random_state=0).fit(XX)
    GMMs.append(gmm)
    


X = np.loadtxt(open("TestSamples-17.csv","rb"),delimiter=",",skiprows=0)
Y = np.loadtxt(open("TestLabels-17.csv","rb"),delimiter=",",skiprows=0)

for i in [0,1,2,3,4,5,6,7,8,9]:
    YY = GMMs[i].score_samples(X)
    if(i==0):
       p = [YY]
    else:
        p = np.row_stack((p,YY))

Label = np.argmax(p, axis=0)

Error = np.nonzero(Y-Label)

print( "Error rate: %f%%\n" %(np.size(Error)*100/len(Y)))
