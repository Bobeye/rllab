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
from lasagne.updates import get_or_compute_grads
from lasagne import utils
from collections import OrderedDict

max_sub_iter = 20

def unpack(i_g):
    i_g_arr = [np.array(x) for x in i_g]
    res = i_g_arr[0].reshape(i_g_arr[0].shape[0]*i_g_arr[0].shape[1])
    res = np.concatenate((res,i_g_arr[1]))
    res = np.concatenate((res,i_g_arr[2][0]))
    res = np.concatenate((res,i_g_arr[3]))
    return res

def adam_svrg(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = []
    updates = []
    updates_of = []
    grads_adam = []
    for m_r in range(2):
        t_prev.append(theano.shared(utils.floatX(0.)))
        updates.append(OrderedDict())
#        grads_adam.append([TT.matrix('eval_grad0'),TT.vector('eval_grad1'),TT.col('eval_grad3'),TT.vector('eval_grad4')])
#        norm_adam.append([TT.matrix('eval_grad0'),TT.vector('eval_grad1'),TT.col('eval_grad3'),TT.vector('eval_grad4')])
        updates_of.append(OrderedDict())
        # Using theano constant to prevent upcasting of float32
        one = TT.constant(1)
        t = t_prev[-1] + 1
        if (m_r==0):
            a_t = learning_rate*TT.sqrt(one-beta2**t)/(one-beta1**t)
        else:
            beta2 = 0.999
            a_t = learning_rate/2*TT.sqrt(one-beta2**t)/(one-beta1**t)
        i = 0
        l = []
        h = []
        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
    
            m_t = beta1*m_prev + (one-beta1)*g_t
            v_t = beta2*v_prev + (one-beta2)*g_t**2
            step = a_t*m_t/(TT.sqrt(v_t) + epsilon)
#            eff_step = TT.sum(TT.square(step,None))
            h.append(TT.sum(TT.square(step)))
            l.append(TT.sum(TT.square(m_t)))
            updates[-1][m_prev] = m_t
            updates[-1][v_prev] = v_t
            updates_of[-1][param] = param - step
            i+=1
    
        updates[-1][t_prev[-1]] = t
        grads_adam.append(TT.sqrt((h[0]+h[1]+h[2]+h[3]+h[4])/(l[0]+l[1]+l[2]+l[3]+l[4])))
    return updates_of,grads_adam
    
def dis_iw(iw):
    z=list()
    t=1
    for y in iw:
        z.append(y*t)
        t*=discount
    return np.array(z)
    
load_policy=True
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(CartpoleEnv())
#env = GymEnv("InvertedPendulum-v1")
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(100,50,25))
snap_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(100,50,25))
back_up_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(100,50,25))
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
T = 500
#We will collect M secondary trajectories
M = 33
#Number of sub-iterations
#m_itr = 100
# Number of iterations
#n_itr = np.int(10000/(m_itr*M+N))
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
#learning_rate = 0.01
learning_rate = 0.001

s_tot = 50000

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
eval_grad5 = TT.matrix('eval_grad3',dtype=grad[2].dtype)
eval_grad6 = TT.vector('eval_grad4',dtype=grad[3].dtype)
eval_grad7 = TT.matrix('eval_grad3',dtype=grad[2].dtype)
eval_grad8 = TT.vector('eval_grad4',dtype=grad[3].dtype)
eval_grad9 = TT.vector('eval_grad4',dtype=grad[3].dtype)

surr_on1 = TT.sum(dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var*importance_weights_var)
surr_on2 = TT.sum(-snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var)
grad_imp = theano.grad(surr_on1,snap_params)

update,step =adam_svrg([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7, eval_grad8, eval_grad9], params, learning_rate=learning_rate)


f_train = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)

f_update = [theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7, eval_grad8, eval_grad9],
    outputs = step[n_sub_iter],
    updates = update[n_sub_iter]
) for n_sub_iter in range(2)]
    
f_importance_weights = theano.function(
    inputs = [observations_var, actions_var],
    outputs = importance_weights
)

f_update_SVRG = [theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7, eval_grad8, eval_grad9],
    outputs = step[n_sub_iter],
    updates = update[n_sub_iter]
) for n_sub_iter in range(2)]

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
ar_data = {}
all_policy_param_data = {}
parallel_sampler.initialize(4)
for k in range(5):
    if (load_policy):
#        snap_policy.set_param_values(np.loadtxt('policy.txt'), trainable=True)
#        policy.set_param_values(np.loadtxt('policy.txt'), trainable=True)
        snap_policy.set_param_values(np.loadtxt('pcb' + np.str(k+1) + '.txt'), trainable=True)
        policy.set_param_values(np.loadtxt('pcb' + np.str(k+1) + '.txt'), trainable=True)
    else:
        policy.set_param_values(snap_policy.get_param_values(trainable=True), trainable=True) 
    avg_return = np.zeros(s_tot)
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    n_sub_iter=[]
    rewards_sub_iter=[]
    rewards_snapshot=[]
    importance_weights=[]
    variance_svrg = []
    variance_sgd = []
    all_rew = []
    all_policy_param = []
    j=0
    while j<s_tot-N:
        all_policy_param.append(policy.get_param_values())
        paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),N,T,show_bar=False)
        paths = paths[:N]
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
        stp_snp = f_update[0](s_g[0],s_g[1],s_g[2],s_g[3],s_g[4],s_g[5],s_g[6],s_g[7],s_g[8])
        
        print("step snapshot:", stp_snp)
        rewards_snapshot.append(np.array([sum(p["rewards"]) for p in paths])) 
        avg_return[j-N:j] = np.repeat(np.mean([sum(p["rewards"]) for p in paths]),N)
        var_4_fg = np.cov(s_g_fv,rowvar=False)
        var_fg = var_4_fg/(N)
        
        all_rew.extend([sum(p["rewards"]) for p in paths])
        print(str(j)+' Snapshot Average Return:', np.mean(all_rew))
        
        back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
        n_sub = 0
        while j<s_tot-M:
            if ((j%100)<M):
                all_policy_param.append(policy.get_param_values())
            j += M
            sub_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),M,T,show_bar=False)
            sub_paths = sub_paths[:M]
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
            w_cum=np.sum(dis_iw(iw_var))
            for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
                i_g_sgd = f_train(ob, ac, rw)
                s_g_fv_sgd.append(unpack(i_g_sgd))
                s_g_sgd = [sum(x) for x in zip(s_g_sgd,i_g_sgd)]
                iw_var = f_importance_weights(ob, ac)
                s_g_is_sgd = f_imp_SVRG(ob, ac, rw,iw_var)
                s_g_fv_is.append(unpack(s_g_is_sgd))
                s_g_is = [sum(x) for x in zip(s_g_is,s_g_is_sgd)] 
                w_cum+=np.sum(dis_iw(iw_var))
            w_cum=len(sub_paths)
            s_g_is = [x/w_cum for x in s_g_is]
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
            all_rew.extend([sum(p["rewards"]) for p in sub_paths])
            rewards_sub_iter.append(np.array([sum(p["rewards"]) for p in sub_paths]))
            avg_return[j-M:j] = np.repeat(np.mean([sum(p["rewards"]) for p in sub_paths]),M)                     
            back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
            g = [sum(x) for x in zip(s_g_is,s_g_sgd,s_g)]  
            stp = f_update[1](g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8])
            if (stp/M<stp_snp/N or n_sub+1>= max_sub_iter):
                break
        n_sub_iter.append(n_sub)
        snap_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True)    
        
    plt.plot(avg_return[::10])
    plt.show()
    rewards_subiter_data["rewardsSubIter"+str(k)]=rewards_sub_iter
    rewards_snapshot_data["rewardsSnapshot"+str(k)]= rewards_snapshot
    n_sub_iter_data["nSubIter"+str(k)]= n_sub_iter
    variance_sgd_data["variancceSgd"+str(k)] = variance_sgd
    variance_svrg_data["varianceSvrg"+str(k)]=variance_svrg
    importance_weights_data["importanceWeights"+str(k)] = importance_weights
    ar_data["mean"+str(k)] = np.mean(all_rew)
    all_policy_param_data["policyParams"+str(k)] = all_policy_param
    

    avg_return=np.array(avg_return)
    #plt.plot(avg_return)
    #plt.show()
    alla["avgReturn"+str(k)]=avg_return
    print("Fine: ",str(k))

alla = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in alla.items() ]))
rewards_subiter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_subiter_data.items() ]))
rewards_snapshot_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_snapshot_data.items() ]))
n_sub_iter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in n_sub_iter_data.items() ]))
variance_sgd_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_sgd_data.items() ]))
variance_svrg_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_svrg_data.items() ]))
importance_weights_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in importance_weights_data.items() ]))
ar_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in ar_data.items() ]))
all_policy_param_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_policy_param_data.items() ]))

rewards_subiter_data.to_csv("rewards_subiter_v2_Vb2.csv",index=False)
rewards_snapshot_data.to_csv("rewards_snapshot_v2_Vb2.csv",index=False)
n_sub_iter_data.to_csv("n_sub_iter_v2_Vb2.csv",index=False)
variance_sgd_data.to_csv("variance_sgd_v2_Vb2.csv",index=False)
variance_svrg_data.to_csv("variance_svrg_v2_Vb2.csv",index=False)
importance_weights_data.to_csv("importance_weights_v2_Vb2.csv",index=False)
ar_data.to_csv("ar_va_b2.csv",index=False)
all_policy_param_data.to_csv("param_policy_v2_Vb2.csv",index=False)