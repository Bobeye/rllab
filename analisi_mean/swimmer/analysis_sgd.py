import numpy as np
import pandas as pd


def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

#load the data

rewards=pd.read_csv("rewards_snapshot_swimmer_mountain_sgd.csv")

#analize
#x=rewards_snapshot["rewardsSnapshot0"] np.array(x[0][1:-1].split()) crea un array contenete i rewards nel primo snapshot


#per trajectories analysis
avg_traj_rewards=list()
for col_name in rewards:    
    traj_rewards = list()
    for s in range(len(rewards[col_name])):
        if (rewards[col_name][s]==rewards[col_name][s]):
            traj_rewards.append(list(rewards[col_name][s][1:-1].split()))
    traj_rewards2=flatten(traj_rewards[:500])
    traj_rewards2.extend(flatten(traj_rewards[500:]))
    traj_rewards3 = np.array(traj_rewards2,dtype = np.float)
    avg_traj_rewards.append(np.mean(traj_rewards3))

np.savetxt('mean_swimmer_sgd.txt',avg_traj_rewards)