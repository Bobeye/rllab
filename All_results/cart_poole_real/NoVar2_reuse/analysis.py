import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

#load the data

n_sub_iter=pd.read_csv("n_sub_iter_adam.csv")
rewards_snapshot=pd.read_csv("rewards_snapshot_adam.csv")
rewards_subiter=pd.read_csv("rewards_subiter_adam.csv")
variance_sgd=pd.read_csv("variance_sgd_adam.csv")
variance_svrg=pd.read_csv("variance_svrg_adam.csv")
importance_weights=pd.read_csv("importance_weights_adam.csv")
gpomdp_rewards=pd.read_csv("GPOMDP_rewards_adam.csv")

#analize
#x=rewards_snapshot["rewardsSnapshot0"] np.array(x[0][1:-1].split()) crea un array contenete i rewards nel primo snapshot


#per trajectories analysis all
avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(list(rewards_snapshot[col_name_s][s][1:-1].split()))
		for j in range(np.int(k)):
			traj_rewards.append(list(rewards_subiter[col_name_si][i+j][1:-1].split()))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
	traj_rewards=flatten(traj_rewards)
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward"],loc="lower right")
plt.savefig("per_tra_analisis_all.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectories analysis
avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(list(rewards_snapshot[col_name_s][s][1:-1].split()))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
	traj_rewards=flatten(traj_rewards)
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward"],loc="lower right")
plt.savefig("per_tra_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per update analysis all
avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][s][1:-1].split(),dtype=np.float64)))
		for j in range(np.int(k)):
			traj_rewards.append(np.mean(np.asarray(rewards_subiter[col_name_si][i+j][1:-1].split(),dtype=np.float64)))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split(),dtype=np.float64)))
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward"],loc="lower right")
plt.savefig("per_upd_analisis_all.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per update analysis
avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][s][1:-1].split(),dtype=np.float64)))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split(),dtype=np.float64)))
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward"],loc="lower right")
plt.savefig("per_upd_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectory comparison all

reward=list()
for col in gpomdp_rewards:
    x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
    rew=list()
    for i in x: 
        rew.append(list(i[1:-1].split()))
    reward.append(flatten(rew[0:500])+flatten(rew[500:]))
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)


avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):    
    first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
    ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
    traj_rewards = list()
    for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
        traj_rewards.append(list(rewards_snapshot[col_name_s][s][1:-1].split()))
        for j in range(np.int(k)):
            traj_rewards.append(list(rewards_subiter[col_name_si][i+j][1:-1].split()))
    if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
        traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
    traj_rewards=flatten(traj_rewards)
    avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
    lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
    temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_tra_comparison_all.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectory comparison 

reward=list()
for col in gpomdp_rewards:
    x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
    rew=list()
    for i in x: 
        rew.append(list(i[1:-1].split()))
    reward.append(flatten(rew[0:500])+flatten(rew[500:]))
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)


avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):    
    first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
    ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
    traj_rewards = list()
    for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
        traj_rewards.append(list(rewards_snapshot[col_name_s][s][1:-1].split()))
    if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
        traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
    traj_rewards=flatten(traj_rewards)
    avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
    lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
    temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_tra_comparison.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectory comparison mean all

reward=list()
for col in gpomdp_rewards:
    x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
    rew=list()
    for i in x: 
        rew.append(list(np.repeat(np.mean(np.array(i[1:-1].split(),dtype = np.float)),10)))
    reward.append(flatten(rew[0:500])+flatten(rew[500:]))
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)


avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):    
    first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
    ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
    traj_rewards = list()
    for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
        traj_rewards.append(list(np.repeat(np.mean(np.array(rewards_snapshot[col_name_s][s][1:-1].split(),dtype = np.float)),100)))
        for j in range(np.int(k)):
             traj_rewards.append(list(np.repeat(np.mean(np.array(rewards_subiter[col_name_si][i+j][1:-1].split(),dtype = np.float)),10)))
#             traj_rewards.append(list(rewards_subiter[col_name_si][i+j][1:-1].split()))
    if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
        traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
    traj_rewards=flatten(traj_rewards)
    avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
    lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
    temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_tra_comparison_mean_all.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectory comparison mean

reward=list()
for col in gpomdp_rewards:
    x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
    rew=list()
    for i in x: 
        rew.append(list(np.repeat(np.mean(np.array(i[1:-1].split(),dtype = np.float)),10)))
    reward.append(flatten(rew[0:500])+flatten(rew[500:]))
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)


avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):    
    first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
    ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
    traj_rewards = list()
    for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
        traj_rewards.append(list(np.repeat(np.mean(np.array(rewards_snapshot[col_name_s][s][1:-1].split(),dtype = np.float)),100)))
    if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
        traj_rewards.append(list(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split()))
    traj_rewards=flatten(traj_rewards)
    avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
    lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
    temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_tra_comparison_mean.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per update comparison all
reward=list()
for col in gpomdp_rewards:
	x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
	rew=list()
	for i in x: 
		rew.append(np.mean(np.array(i[1:-1].split(),dtype=np.float64)))
	reward.append(rew)
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)

avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][s][1:-1].split(),dtype=np.float64)))
		for j in range(np.int(k)):
			traj_rewards.append(np.mean(np.asarray(rewards_subiter[col_name_si][i+j][1:-1].split(),dtype=np.float64)))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split(),dtype=np.float64)))
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_update_analisis_all.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per update comparison
reward=list()
for col in gpomdp_rewards:
	x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
	rew=list()
	for i in x: 
		rew.append(np.mean(np.array(i[1:-1].split(),dtype=np.float64)))
	reward.append(rew)
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)

avg_traj_rewards=list()
for col_name_s,col_name_si,col_name_nsi in zip(rewards_snapshot,rewards_subiter,n_sub_iter):	
	first = np.insert(np.array(np.cumsum(n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])])),0,0) #np.cumsum(n_sub_iter["nSubIter0"][~np.isnan(n_sub_iter["nSubIter0"])])
	ranges = n_sub_iter[col_name_nsi][~np.isnan(n_sub_iter[col_name_nsi])]
	traj_rewards = list()
	for i,k,s in zip(first[0:-1],ranges,range(len(ranges))):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][s][1:-1].split(),dtype=np.float64)))
	if(len(ranges)<len(rewards_snapshot[col_name_s][rewards_snapshot[col_name_s]==rewards_snapshot[col_name_s]])):
		traj_rewards.append(np.mean(np.asarray(rewards_snapshot[col_name_s][np.int(s+1)][1:-1].split(),dtype=np.float64)))
	avg_traj_rewards.append(traj_rewards)
lenghts=list()
for i in avg_traj_rewards:
	lenghts.append(len(i))
min_len=min(lenghts)
temp=list()
for i in avg_traj_rewards:
	temp.append(i[:min_len])
avg_traj_rewards=temp

avg_traj_rewards=np.asarray(np.mean(np.matrix(avg_traj_rewards,dtype=np.float64),axis=0)).flatten()
plt.plot(avg_traj_rewards)
plt.legend(["Average Reward SGD","Average Reward SVRG"],loc="lower right")
plt.savefig("per_update_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

# variance sgd analysis
variance_sgd_l=list()
for col in variance_sgd:
	variance_sgd_l.append(variance_sgd[col][variance_sgd[col]==variance_sgd[col]])
lenghts=list()
for i in variance_sgd_l:
	lenghts.append(len(i))
min_len=min(lenghts)
variance_sgd=list()
for i in variance_sgd_l:
	variance_sgd.append(i[:min_len])
variance_sgd=np.asarray(np.mean(np.matrix(variance_sgd,dtype=np.float64),axis=0)).flatten()
plt.plot(variance_sgd)


#variance svrg analysis
variance_svrg_l=list()
for col in variance_svrg:
	variance_svrg_l.append(variance_svrg[col][variance_svrg[col]==variance_svrg[col]])
lenghts=list()
for i in variance_svrg_l:
	lenghts.append(len(i))
min_len=min(lenghts)
variance_svrg=list()
for i in variance_svrg_l:
	variance_svrg.append(i[:min_len])
variance_svrg=np.asarray(np.mean(np.matrix(variance_svrg,dtype=np.float64),axis=0)).flatten()
plt.plot(variance_svrg)
legend=["SGD Variance","SVRG Variance"]
plt.legend(legend,loc="upper left")
plt.savefig("var_svrg_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

#sub iterations analysis
sub_iter_l=list()
for col in n_sub_iter:
	sub_iter_l.append(n_sub_iter[col][n_sub_iter[col]==n_sub_iter[col]])
lenghts=list()
for i in sub_iter_l:
	lenghts.append(len(i))
min_len=min(lenghts)
n_sub_iter=list()
for i in sub_iter_l:
	n_sub_iter.append(i[:min_len])
n_sub_iter=np.asarray(np.mean(np.matrix(n_sub_iter,dtype=np.float64),axis=0)).flatten()
plt.plot(n_sub_iter)
plt.legend(["Sub Iterations Performed"],loc="upper right")
plt.savefig("sub_iter_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

#importance weights analysis
i_w_list=list()
for col in importance_weights:
	i_w_list.append(importance_weights[col][importance_weights[col]==importance_weights[col]])
lenghts=list()
for i in i_w_list:
	lenghts.append(len(i))
min_len=min(lenghts)
importance_weights=list()
for i in i_w_list:
	importance_weights.append(i[:min_len])
importance_weights=np.asarray(np.mean(np.matrix(importance_weights,dtype=np.float64),axis=0)).flatten()
plt.plot(importance_weights)
plt.legend(["Average Importance Weights"],loc="upper right")
plt.savefig("imp_we_analisis.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per trajectory analysis GPOMDP
reward=list()
for col in gpomdp_rewards:
	x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
	rew=list()
	for i in x: 
		rew.append(list(i[1:-1].split()))
	reward.append(flatten(rew[0:500])+flatten(rew[500:]))
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)
plt.legend(["Average Reward GPOMDP"],loc="lower right")
plt.savefig("per_tra_analisis_GPOMDP.jpg", figsize=(32, 24), dpi=160)
plt.show()

#per update analysis GPOMDP
reward=list()
for col in gpomdp_rewards:
	x=gpomdp_rewards[col][gpomdp_rewards[col]==gpomdp_rewards[col]]
	rew=list()
	for i in x: 
		rew.append(np.mean(np.array(i[1:-1].split(),dtype=np.float64)))
	reward.append(rew)
reward=np.asarray(np.mean(np.matrix(reward,dtype=np.float64),axis=0)).flatten()
plt.plot(reward)
plt.legend(["Average Reward GPOMDP"],loc="lower right")
plt.savefig("per_update_GPOMDP.jpg", figsize=(32, 24), dpi=160)
plt.show()
