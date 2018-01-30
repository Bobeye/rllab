import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

svrpgRewards=pd.read_csv("interConf/rewardsparam_policy_v2_V.csv")
gpomdpRewards=pd.read_csv("interConf/rewardsparam_policy_GPOMDP500.csv")
nIterations=1000
alpha=0.05
k=100
H=[]
L=[]
M=[]
for j,row in svrpgRewards.iterrows():
	stat=[]
	for i in range(nIterations):
		stat.append(np.mean(np.random.choice(np.array(row),size=np.int(len(row)*0.5),replace=True)))
	p=alpha/2/k*100
	L.append(max(0.0, np.percentile(stat, p)))
	p=(1-alpha/2/k)*100
	H.append(min(1000, np.percentile(stat, p)))
	M.append(np.mean(stat))
plt.figure(figsize=(9,9))
plt.title('Cart-Pole Task: SVRPG vs GPOMDP')
plt.xlabel('Trajectories')
plt.ylabel('Average Return')
m=len(H)
x = np.linspace(0, 100*m, num=m,endpoint=False)
f=interp1d(x, M)
plt.plot(x,f(x),color="#ff0000")
plt.plot(x,M,"o",label="Avg Return SVRPG",alpha=0.5,color="#478fc1")
ax = plt.gca()
H=np.asarray(H,dtype=np.float64)
L=np.asarray(L,dtype=np.float64)
ax.vlines(x, L, H,alpha="0.5",color="#478fc1")
H=[]
L=[]
M=[]
for j,row in gpomdpRewards.iterrows():
	stat=[]
	for i in range(nIterations):
		stat.append(np.mean(np.random.choice(np.array(row),size=np.int(len(row)*0.5),replace=True)))
	p=alpha/2/k*100
	L.append(max(0.0, np.percentile(stat, p)))
	p=(1-alpha/2/k)*100
	H.append(min(1000, np.percentile(stat, p)))
	M.append(np.mean(stat))
m=len(H)
x = np.linspace(0, 100*m, num=m,endpoint=False)
f=interp1d(x, M)
plt.plot(x,f(x),color="#ff0000")
plt.plot(x,M,"o",label="Avg Return GPOMDP",alpha=0.5,color="#ff8a25")
H=np.asarray(H,dtype=np.float64)
L=np.asarray(L,dtype=np.float64)
ax.vlines(x, L, H,alpha="0.5",color="#ff8a25")
plt.legend(loc="lower right")

plt.show()

svrpgRewards=np.asmatrix(svrpgRewards)
for i in range(nIterations):
	stat.append(np.mean(np.random.choice(svrpgRewards,size=svrpgRewards.dim[0]*svrpgRewards.dim[0]*0.5,replace=True)))
p=alpha/2/k*100
L.append(max(0.0, np.percentile(stat, p)))
p=(1-alpha/2/k)*100
H.append(min(1000, np.percentile(stat, p)))
M.append(np.mean(stat))