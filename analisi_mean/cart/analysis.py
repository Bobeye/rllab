import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

#load the data

rewards_snapshot=pd.read_csv("rewards_snapshot.csv")
rewards_subiter=pd.read_csv("rewards_subiter.csv")

#analize
#x=rewards_snapshot["rewardsSnapshot0"] np.array(x[0][1:-1].split()) crea un array contenete i rewards nel primo snapshot


#per trajectories analysis
avg_traj_rewards=list()
for col_name_s,col_name_si in zip(rewards_snapshot,rewards_subiter):    
    traj_rewards = list()
    for s in range(len(rewards_snapshot[col_name_s])):
        if (rewards_snapshot[col_name_s][s]==rewards_snapshot[col_name_s][s]):
            traj_rewards.append(list(rewards_snapshot[col_name_s][s][1:-1].split()))
    for s in range(len(rewards_subiter[col_name_si])):
        if (rewards_subiter[col_name_si][s]==rewards_subiter[col_name_si][s]):
            traj_rewards.append(list(rewards_subiter[col_name_si][s][1:-1].split()))
    traj_rewards2=flatten(traj_rewards)
    traj_rewards3 = np.array(traj_rewards2,dtype = np.float)
    avg_traj_rewards.append(np.mean(traj_rewards3))

np.savetxt('mean_svrg.txt',avg_traj_rewards)