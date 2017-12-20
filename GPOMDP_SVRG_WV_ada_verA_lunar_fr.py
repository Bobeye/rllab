from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import sgd
import matplotlib.pyplot as plt
from rllab.envs.gym_env import GymEnv
import pandas as pd

def unpack(i_g):
    i_g_arr = [np.array(x) for x in i_g]
    res = i_g_arr[0].reshape(i_g_arr[0].shape[0]*i_g_arr[0].shape[1])
    res = np.concatenate((res,i_g_arr[1]))
    res = np.concatenate((res,i_g_arr[2][0]))
    res = np.concatenate((res,i_g_arr[3]))
    return res
    
    
load_policy=True
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(GymEnv("LunarLanderContinuous-v2"))
#env = GymEnv("InvertedPendulum-v1")
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
snap_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
back_up_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
parallel_sampler.populate_task(env, policy)

# policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
# distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
# the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
# rllab.distributions.DiagonalGaussian
dist = policy.distribution
snap_dist = snap_policy.distribution
# We will collect 100 trajectories per iteration
N = 100
# Each trajectory will have at most 100 time steps
T = 1000
#We will collect M secondary trajectories
M = 10
#Number of sub-iterations
#m_itr = 100
# Number of iterations
#n_itr = np.int(10000/(m_itr*M+N))
# Set the discount factor for the problem
discount = 0.995
# Learning rate for the gradient update
learning_rate = 0.00005

s_tot = 10000

observations_var = env.observation_space.new_tensor_variable(
    'observations',
    # It should have 1 extra dimension since we want to represent a list of observations
    extra_dims=1
)
actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)
d_rewards_var = TT.vector('d_rewards')
importance_weights_var = TT.vector('importance_weight')

# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)
snap_dist_info_vars = snap_policy.dist_info_sym(observations_var)

surr = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)

params = policy.get_params(trainable=True)
snap_params = snap_policy.get_params(trainable=True)

importance_weights = dist.likelihood_ratio_sym_1traj_GPOMDP(actions_var,dist_info_vars,snap_dist_info_vars)

grad = theano.grad(surr, params)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.matrix('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

surr_on1 = TT.sum(dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var*importance_weights_var)
surr_on2 = TT.sum(-snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var)
grad_imp = theano.grad(surr_on1,snap_params)

f_train = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)
f_update = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
    outputs = None,
    updates = sgd([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params, learning_rate=learning_rate)
)
f_importance_weights = theano.function(
    inputs = [observations_var, actions_var],
    outputs = importance_weights
)

f_update_SVRG = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
    outputs = None,
    updates = sgd([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params, learning_rate=learning_rate)
)

f_imp_SVRG = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var, importance_weights_var],
    outputs=grad_imp,
)

alla = {}
variance_svrg_data={}
variance_sgd_data={}
importance_weights_data={}
rewards_snapshot_data={}
rewards_subiter_data={}
n_sub_iter_data={}
for k in range(10):
    if (load_policy):
        snap_policy.set_param_values(np.loadtxt('policy_lunar_novar.txt'), trainable=True)
        policy.set_param_values(np.loadtxt('policy_lunar_novar.txt'), trainable=True)
    else:
        policy.set_param_values(snap_policy.get_param_values(trainable=True), trainable=True) 
    avg_return = []
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    n_sub_iter=[]
    rewards_sub_iter=[]
    rewards_snapshot=[]
    importance_weights=[]
    variance_svrg = []
    variance_sgd = []
    j=0
    while j<s_tot-N:
        paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),N,T,show_bar=False)
        #baseline.fit(paths)
        j+=N
        observations = [p["observations"] for p in paths]
        actions = [p["actions"] for p in paths]
        d_rewards = [p["rewards"] for p in paths]
        temp = list()
        for x in d_rewards:
            z=list()
            t=1
            for y in x:
                z.append(y*t)
                t*=discount
            temp.append(np.array(z))
        d_rewards=temp
        s_g = f_train(observations[0], actions[0], d_rewards[0])
        s_g_fv = [unpack(s_g)]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            i_g = f_train(ob, ac, rw)
            s_g_fv.append(unpack(i_g))
            s_g = [sum(x) for x in zip(s_g,i_g)]
        s_g = [x/len(paths) for x in s_g]
        
        f_update(s_g[0],s_g[1],s_g[2],s_g[3])
        rewards_snapshot.append(np.array([sum(p["rewards"]) for p in paths])) 
        avg_return.append(np.mean([sum(p["rewards"]) for p in paths]))
    
        var_4_fg = np.cov(s_g_fv,rowvar=False)
        var_fg = var_4_fg/(N)
        
        print(str(j-1)+' Average Return:', avg_return[-1])
        
        back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
        n_sub = 0
        while j<s_tot-M:
            j += M
            sub_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),M,T,show_bar=False)
            #baseline.fit(paths)
            sub_observations=[p["observations"] for p in sub_paths]
            sub_actions = [p["actions"] for p in sub_paths]
            sub_d_rewards = [p["rewards"] for p in sub_paths]
            temp = list()
            for x in sub_d_rewards:
                z=list()
                t=1
                for y in x:
                    z.append(y*t)
                    t*=discount
                temp.append(np.array(z)) 
            sub_d_rewards=temp
            n_sub+=1
            s_g_sgd = f_train(sub_observations[0], sub_actions[0], sub_d_rewards[0])
            s_g_fv_sgd = [unpack(s_g_sgd)]
            iw_var = f_importance_weights(sub_observations[0], sub_actions[0])
            s_g_is = f_imp_SVRG(sub_observations[0], sub_actions[0], sub_d_rewards[0],iw_var)
            s_g_fv_is = [unpack(s_g_is)]
            for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
                i_g_sgd = f_train(ob, ac, rw)
                s_g_fv_sgd.append(unpack(i_g_sgd))
                s_g_sgd = [sum(x) for x in zip(s_g_sgd,i_g_sgd)]
                iw_var = f_importance_weights(ob, ac)
                s_g_is_sgd = f_imp_SVRG(ob, ac, rw,iw_var)
                s_g_fv_is.append(unpack(s_g_is_sgd))
                s_g_is = [sum(x) for x in zip(s_g_is,s_g_is_sgd)] 
            s_g_is = [x/len(sub_paths) for x in s_g_is]
            s_g_sgd = [x/len(sub_paths) for x in s_g_sgd]
            var_sgd = np.cov(s_g_fv_sgd,rowvar=False)
            var_batch = var_sgd/(M)
            var_is_sgd = np.cov(s_g_fv_is,rowvar=False)
            var_is = var_is_sgd/(M)
            m_is = np.mean(s_g_fv_is,axis=0)
            m_sgd = np.mean(s_g_fv_sgd,axis=0)
            cov= np.outer(s_g_fv_is[0]-m_is,s_g_fv_sgd[0]-m_sgd)
            for i in range(M-1):
              cov += np.outer(s_g_fv_is[i+1]-m_is,s_g_fv_sgd[i+1]-m_sgd)  
            for i in range(M):
              cov += np.outer(s_g_fv_sgd[i]-m_sgd,s_g_fv_is[i]-m_is)  
            cov = cov/(M*M)
            var_svrg = var_fg + var_is + var_batch + cov
            var_dif = var_svrg-var_batch
            iw = f_importance_weights(sub_observations[0],sub_actions[0])
            importance_weights.append(np.mean(iw))
            variance_svrg.append((np.diag(var_svrg).sum()))
            variance_sgd.append((np.diag(var_batch).sum()))
            rewards_sub_iter.append(np.array([sum(p["rewards"]) for p in sub_paths]))
            avg_return.append(np.mean([sum(p["rewards"]) for p in sub_paths]))
            if (np.trace(var_dif)>0):
                policy.set_param_values(back_up_policy.get_param_values(trainable=True), trainable=True) 
                break            
            
            back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
            g = [sum(x) for x in zip(s_g_is,s_g_sgd,s_g)]  
            f_update(g[0],g[1],g[2],g[3])
            #print(str(j)+' Average Return:', avg_return[j])
        n_sub_iter.append(n_sub)
        snap_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True)    
        
#    plt.plot(avg_return)
#    plt.show()
    rewards_subiter_data["rewardsSubIter"+str(k)]=rewards_sub_iter
    rewards_snapshot_data["rewardsSnapshot"+str(k)]= rewards_snapshot
    n_sub_iter_data["nSubIter"+str(k)]= n_sub_iter
    variance_sgd_data["variancceSgd"+str(k)] = variance_sgd
    variance_svrg_data["varianceSvrg"+str(k)]=variance_svrg
    importance_weights_data["importanceWeights"+str(k)] = importance_weights

    avg_return=np.array(avg_return)
    #plt.plot(avg_return)
    #plt.show()
    alla["avgReturn"+str(k)]=avg_return

alla = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in alla.items() ]))
rewards_subiter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_subiter_data.items() ]))
rewards_snapshot_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_snapshot_data.items() ]))
n_sub_iter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in n_sub_iter_data.items() ]))
variance_sgd_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_sgd_data.items() ]))
variance_svrg_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_svrg_data.items() ]))
importance_weights_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in importance_weights_data.items() ]))

rewards_subiter_data.to_csv("rewards_subiter_vA_lunar.csv",index=False)
rewards_snapshot_data.to_csv("rewards_snapshot_vA_lunar.csv",index=False)
n_sub_iter_data.to_csv("n_sub_iter_v2_lunar.csv",index=False)
variance_sgd_data.to_csv("variance_sgd_vA_lunar.csv",index=False)
variance_svrg_data.to_csv("variance_svrg_vA_lunar.csv",index=False)
importance_weights_data.to_csv("importance_weights_vA_lunar.csv",index=False)

alla.to_csv("GPOMDP_SVRG_adaptive_m06_verA_lunar.csv",index=False)