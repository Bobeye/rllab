import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

gpomdpRewards100=pd.read_csv("interConf/rewardsparam_policy_hc_sgd100.csv")
gpomdpRewards10=pd.read_csv("rewardsGPOM_cartPole_100_nonself.csv")
snSvrpgRewards=pd.read_csv("rewards_cartPole_100_nonself.csv")#("interConf/rewardsparam_policy_GPOMDP500.csv")#("interConf/cartpole100/rewardsGPOM_cartPole_100_nonself.csv")
nIterations=1000
alpha=0.1
maxReward=2000
minReward=-360
m=1000

for col in snSvrpgRewards:
	if(len(snSvrpgRewards[col][snSvrpgRewards[col]==snSvrpgRewards[col]])<m):
		m=len(snSvrpgRewards[col][snSvrpgRewards[col]==snSvrpgRewards[col]])
gpomdpRewards10=gpomdpRewards10[0:m]
gpomdpRewards100=gpomdpRewards100[0:m]
snSvrpgRewards=snSvrpgRewards[0:m]
k=1

H=[]
L=[]
M=[]
for j,row in snSvrpgRewards.iterrows():
	stat=[]
	for i in range(nIterations):
		stat.append(np.mean(np.random.choice(np.array(row),size=np.int(len(row)*1),replace=True)))
	p=alpha/2/k*100
	L.append(max(minReward, np.percentile(stat, p)))
	p=(1-alpha/2/k)*100
	H.append(min(maxReward, np.percentile(stat, p)))
	M.append(np.mean(stat))

plt.xlabel('Trajectories')
plt.ylabel('Average Return')
x = np.linspace(0, 500*m, num=m,endpoint=False)
f=interp1d(x, M)
plt.plot(x,f(x),color="#ff0000")
plt.plot(x,M,"o",label="Avg Return self-normalized SVRPG",alpha=0.5,color="#ff8a25")
H=np.asarray(H,dtype=np.float64)
L=np.asarray(L,dtype=np.float64)

#ax.vlines(x, L, H,alpha="0.5",color="#ff8a25")
plt.fill_between(x,H,L,color="#ff8a25",alpha="0.3")

H=[]
L=[]
M=[]
for j,row in gpomdpRewards10.iterrows():
	stat=[]
	for i in range(nIterations):
		stat.append(np.mean(np.random.choice(np.array(row),size=np.int(len(row)*1),replace=True)))
	p=alpha/2/k*100
	L.append(max(minReward, np.percentile(stat, p)))
	p=(1-alpha/2/k)*100
	H.append(min(maxReward, np.percentile(stat, p)))
	M.append(np.mean(stat))

x = np.linspace(0, 500*m, num=m,endpoint=False)
f=interp1d(x, M)
plt.plot(x,f(x),color="green")
plt.plot(x,M,"o",label="Avg Return GPOMDP 10",alpha=0.5,color="#58c147")
H=np.asarray(H,dtype=np.float64)
L=np.asarray(L,dtype=np.float64)

H=[]
L=[]
M=[]
for j,row in gpomdpRewards100.iterrows():
	stat=[]
	for i in range(nIterations):
		stat.append(np.mean(np.random.choice(np.array(row),size=np.int(len(row)*1),replace=True)))
	p=alpha/2/k*100
	L.append(max(minReward, np.percentile(stat, p)))
	p=(1-alpha/2/k)*100
	H.append(min(maxReward, np.percentile(stat, p)))
	M.append(np.mean(stat))

x = np.linspace(0, 500*m, num=m,endpoint=False)
f=interp1d(x, M)
plt.plot(x,f(x),color="red")
plt.plot(x,M,"o",label="Avg Return GPOMDP 100",alpha=0.5,color="red")
H=np.asarray(H,dtype=np.float64)
L=np.asarray(L,dtype=np.float64)

#ax.vlines(x, L, H,alpha="0.5",color="#58c147")
plt.fill_between(x,H,L,color="#e5451d",alpha="0.3")

plt.legend(loc="lower right")
plt.savefig("asda.pdf", format='pdf')
plt.show()

#svrpgRewards=np.asarray(svrpgRewards.sum().values)
#stat=[]
#for i in range(nIterations):
#	stat.append(np.mean(np.random.choice(svrpgRewards,size=np.int(svrpgRewards.size*0.5),replace=True)))
#p=alpha/2*100
#max(0.0, np.percentile(stat, p))
#p=(1-alpha/2)*100
#min(100000, np.percentile(stat, p))
#np.mean(stat)