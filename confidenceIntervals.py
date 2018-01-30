from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
from rllab.sampler import parallel_sampler
import pandas as pd
from scipy.stats import t
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=True)

parallel_sampler.populate_task(env, policy)
parallel_sampler.initialize(8)

T=100
N=500
file=["param_policy_GPOMDP500.csv","param_policy_v2_V.csv"]
lab=["SVRPG","GPOMDP"]
plt.figure(figsize=(9,9))
plt.title('Cart-Pole Task: SVRPG vs GPOMDP')
plt.xlabel('Trajectories')
plt.ylabel('Average Return')

for f,l in zip(file,lab):
	print("Processing file "+f+" ...")
	rewards={}
	params=pd.read_csv(f)
	m=100
	for col in params:
		if(len(params[col][params[col]==params[col]])<m):
			m=len(params[col][params[col]==params[col]])
	i=0
	for col in params:
		r=[]
		print("Processing "+col+" ...")
		for s in params[col][:m]:
			param=np.asarray(s[1:-1].split(),dtype=np.float64)
			policy.set_param_values(param, trainable=True)
			paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),N,T,show_bar=False)
			r.append(np.mean(np.asarray([p["rewards"].sum() for p in paths],dtype=np.float64)))
		rewards["run"+str(i)]=r
		i=i+1
	rewards=pd.DataFrame(rewards)
	alpha=0.05
	k=100
	L=[]
	H=[]
	M=[]
	for i,row in rewards.iterrows():
		mean=np.mean(row)
		std=np.std(row,ddof=1)
		t_bounds = t.interval(1-alpha/k, len(row) - 1)
		ci = [mean + critval * std / np.sqrt(len(row)) for critval in t_bounds]
		L.append(ci[0])
		H.append(ci[1])
		M.append(mean)
	data={}
	data["Low"]=L
	data["High"]=H
	data["Mean"]=M
	data=pd.DataFrame(data)
	data.to_csv("interConf/interConf"+f,index=False)
	rewards.to_csv("interConf/rewards"+f,index=False)
	x = np.linspace(0, 100*m, num=m,endpoint=False)
	f=interp1d(x, M)
	plt.errorbar(x=np.arange(0, m*100, 100), 
	             y=M, 
	             yerr=[(top-bot)/2 for top,bot in zip(H,L)],
	             fmt='o',alpha=0.5,label="Avg Return"+l)
	plt.plot(x,f(x),color="red")

	ax = plt.gca()
	H=np.asarray(H,dtype=np.float64)
	L=np.asarray(L,dtype=np.float64)
	ax.vlines(x[H>1000], [1000], H[H>1000],color="black",alpha="0.5")
	ax.vlines(x[L<0], [0], L[L<0],color="black",alpha="0.5")
plt.legend(loc="lower right")

plt.show()
