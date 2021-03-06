from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import adam
import matplotlib.pyplot as plt
from rllab.envs.gym_env import GymEnv
import pandas as pd
from lasagne.updates import get_or_compute_grads
from lasagne import utils
from collections import OrderedDict

max_sub_iter = 10

def unpack(i_g):
    i_g_arr = [np.array(x) for x in i_g]
    res = i_g_arr[0].reshape(i_g_arr[0].shape[0]*i_g_arr[0].shape[1])
    res = np.concatenate((res,i_g_arr[1]))
    res = np.concatenate((res,i_g_arr[2][0]))
    res = np.concatenate((res,i_g_arr[3]))
    return res



def compute_snap_batch(observations,actions,d_rewards,n_traj,n_part):
    n=n_traj
    i=0
    svrg_snap=list()
    while(n-np.int(n_traj/n_part)>=0):
        n=n-np.int(n_traj/n_part)
        s_g = f_train(observations[i], actions[i], d_rewards[i])
        for s in range(i+1,i+np.int(n_traj/n_part)):
            s_g = [sum(x) for x in zip(s_g,f_train(observations[s], actions[s], d_rewards[s]))]
        s_g = [x/np.int(n_traj/n_part) for x in s_g]
        i += np.int(n_traj/n_part)
        svrg_snap.append(unpack(s_g))
    return svrg_snap

def estimate_variance(observations,actions,d_rewards,snap_grads,n_traj,n_traj_s,n_part,M,N):
    n=n_traj
    i=0
    svrg=list()
    j=0
    while(n-np.int(n_traj/n_part)>=0):
        n=n-np.int(n_traj/n_part)
        iw = f_importance_weights(observations[i],actions[i])
        iw_cum = np.sum(dis_iw(iw))
        x = unpack(f_train_SVRG_4v(observations[i],actions[i],d_rewards[i],iw))*np.sqrt(np.int(n_traj/n_part)/M)
        g = snap_grads[j]*np.sqrt(np.int(n_traj_s/n_part)/N)+x
        for s in range(i+1,i+np.int(n_traj/n_part)):
            iw = f_importance_weights(observations[s],actions[s])
            iw_cum += np.sum(dis_iw(iw))
            g_prov=unpack(f_train_SVRG_4v(observations[s],actions[s],d_rewards[s],iw))*np.sqrt(np.int(n_traj/n_part)/M)
            g+=snap_grads[j]*np.sqrt(np.int((n_traj_s)/n_part)/N) + g_prov
        g=g/n_traj*n_part
        i+=np.int(n_traj/n_part)
        j+=1
        svrg.append(g)
    return (np.diag(np.cov(np.matrix(svrg),rowvar=False)).sum())


def estimate_full_gradient_var(data):
    var_4_fg = np.cov(data,rowvar=False)
    var_fg = var_4_fg/(N)
    return var_fg
    
def estimate_SVRG_and_SGD_var(observations,actions,d_rewards,var_fg):
    s_g_sgd = [-x for x in f_train(observations[0], actions[0], d_rewards[0])]
    s_g_fv_sgd = [unpack(s_g_sgd)]
    iw_var = f_importance_weights(observations[0],actions[0])
    s_g_is = var_SVRG(observations[0], actions[0], d_rewards[0],iw_var)
    s_g_fv_is = [unpack(s_g_is)]
    w_cum=np.sum(dis_iw(iw_var))
    for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
        i_g_sgd = [-x for x in f_train(ob, ac, rw)]
        s_g_fv_sgd.append(unpack(i_g_sgd))
        iw_var = f_importance_weights(ob, ac)
        w_cum+=np.sum(dis_iw(iw_var))
        s_g_is = var_SVRG(ob, ac, rw,iw_var)
        s_g_fv_is.append(unpack(s_g_is))
#    w_cum = M
    var_sgd = np.cov(s_g_fv_sgd,rowvar=False)
    var_batch = var_sgd/(M)
    var_is_sgd = np.cov(s_g_fv_is,rowvar=False)
    var_is = var_is_sgd/(w_cum)
    m_is = np.mean(s_g_fv_is,axis=0)
    m_sgd = np.mean(s_g_fv_sgd,axis=0)
    l=len(observations)
    cov= np.outer(s_g_fv_is[0]-m_is,s_g_fv_sgd[0]-m_sgd)
    for i in range(l-1):
      cov += np.outer(s_g_fv_is[i+1]-m_is,s_g_fv_sgd[i+1]-m_sgd)  
    for i in range(l):
      cov += np.outer(s_g_fv_sgd[i]-m_sgd,s_g_fv_is[i]-m_is)  
    cov = cov/(l*np.sqrt(M*w_cum))
    var_svrg = var_fg + var_is + var_batch + cov
    return var_svrg,var_batch



def adam_svrg(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = []
    updates = []
    updates_of = []
    grads_adam = []
    for m_r in range(max_sub_iter):
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
            a_t = learning_rate/3*TT.sqrt(one-beta2**t)/(one-beta1**t)
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
        grads_adam.append(TT.sqrt((h[0]+h[1]+h[2]+h[3]+h[4]+h[5]+h[6])/(l[0]+l[1]+l[2]+l[3]+l[4]+l[5]+l[6])))
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
env = normalize(GymEnv("Swimmer-v1"))
#env = GymEnv("InvertedPendulum-v1")
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32))
snap_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32))
back_up_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32))
parallel_sampler.populate_task(env, snap_policy)

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
learning_rate = 0.001
#perc estimate
perc_est = 0.6
#tot trajectories
s_tot = 20000

partition = 3


porz = np.int(perc_est*N)

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

importance_weights = dist.likelihood_ratio_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars,dist_info_vars)

grad = theano.grad(surr, params)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.matrix('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)
eval_grad5 = TT.matrix('eval_grad5',dtype=grad[3].dtype)
eval_grad6 = TT.vector('eval_grad5',dtype=grad[3].dtype)
eval_grad7 = TT.vector('eval_grad5',dtype=grad[3].dtype)


surr_on1 = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var*importance_weights_var)
surr_on2 = TT.sum(snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var)
grad_SVRG =theano.grad(surr_on2,snap_params) #,theano.grad(surr_on1,params))]
grad_SVRG_4v = [sum(x) for x in zip(theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
grad_imp = theano.grad(surr_on1,params)
grad_var = theano.grad(surr_on1,params)

update,step =adam_svrg([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5,eval_grad6,eval_grad7], params, learning_rate=learning_rate)


f_train = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)

f_update = [theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5,eval_grad6,eval_grad7],
    outputs = step[n_sub_iter],
    updates = update[n_sub_iter]
) for n_sub_iter in range(max_sub_iter)]

f_importance_weights = theano.function(
    inputs = [observations_var, actions_var],
    outputs = importance_weights
)



f_update_SVRG = [theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5,eval_grad6,eval_grad7],
    outputs = step[n_sub_iter],
    updates = update[n_sub_iter]
) for n_sub_iter in range(max_sub_iter)]



f_train_SVRG = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var],
    outputs=grad_SVRG,
)



f_train_SVRG_4v = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var,importance_weights_var],
    outputs=grad_SVRG_4v,
)



var_SVRG = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var, importance_weights_var],
    outputs=grad_var,
)

f_train_imp = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var, importance_weights_var],
    outputs=grad_var,
)



variance_svrg_data={}
variance_sgd_data={}
importance_weights_data={}
rewards_snapshot_data={}
rewards_subiter_data={}
n_sub_iter_data={}
parallel_sampler.initialize(15)
for k in range(10):
    if (load_policy):
        snap_policy.set_param_values(np.loadtxt('policy_swimmer.txt'), trainable=True)
        policy.set_param_values(np.loadtxt('policy_swimmer.txt'), trainable=True)
    avg_return = list()
    n_sub_iter=[]
    rewards_sub_iter=[]
    rewards_snapshot=[]
    importance_weights=[]
    variance_svrg = []
    variance_sgd = []

    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    j=0
    while j<s_tot-N:
        paths = parallel_sampler.sample_paths_on_trajectories(snap_policy.get_param_values(),N,T,show_bar=False)
        #baseline.fit(paths)
        paths = paths[:N]
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

        full_g_variance=estimate_full_gradient_var(s_g_fv)

        rewards_snapshot.append(np.array([sum(p["rewards"]) for p in paths])) 
        avg_return.append(np.mean([sum(p["rewards"]) for p in paths]))
        stp_snp = np.sum(f_update[0](s_g[0],s_g[1],s_g[2],s_g[3],s_g[4],s_g[5],s_g[6]))
        print(str(j-1)+' Snapshot! Average Return:', avg_return[-1])
        print("step snapshot:", stp_snp)
        

        back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
        n_sub = 0
        sub_ob_acc=list()
        sub_ac_acc=list()
        sub_d_rew_acc=list()
        while j<s_tot-M:
            j += M
            sub_paths = parallel_sampler.sample_paths_on_trajectories(snap_policy.get_param_values(),M,T,show_bar=False)
            #baseline.fit(paths)
            sub_paths = sub_paths[:M]
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
            sub_d_rewards = temp
            sub_ob_acc+=sub_observations
            sub_ac_acc+=sub_actions
            sub_d_rew_acc+=sub_d_rewards
            var_svrg,var_batch=estimate_SVRG_and_SGD_var(sub_ob_acc,sub_ac_acc,sub_d_rew_acc,full_g_variance)
            var_dif = var_svrg-var_batch
            p=snap_policy.get_param_values(trainable=True)
            s_p = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),10,T,show_bar=False)
            s_p = s_p[:M]
            snap_policy.set_param_values(p,trainable=True)
            rewards_sub_iter.append(np.array([sum(p["rewards"]) for p in s_p]))
            avg_return.append(np.mean([sum(p["rewards"]) for p in s_p]))
            print(str(j-1)+' Average Return:', avg_return[-1])
            variance_svrg.append(np.trace(var_svrg))
            variance_sgd.append(np.trace(var_batch))

            #print(np.sum(eigval))
            n_sub+=1
            
            iw = f_importance_weights(sub_observations[0],sub_actions[0])
            importance_weights.append(np.mean(iw))
            #eigval = np.real(np.linalg.eig(var_dif)[0])
            back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
            iw_cum = np.sum(dis_iw(iw))
            g = f_train_SVRG(sub_observations[0],sub_actions[0],sub_d_rewards[0])
            g_is = f_train_imp(sub_observations[0],sub_actions[0],sub_d_rewards[0],iw)
            for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
                iw = f_importance_weights(ob,ac)
                iw_cum += np.sum(dis_iw(iw))
                importance_weights.append(np.mean(iw))
                g_is = [sum(x) for x in zip(g_is,f_train_imp(ob,ac,rw,iw))]
                g = [sum(x) for x in zip(g,f_train_SVRG(ob,ac,rw))]
#            iw_cum = M
            g = [x/len(sub_paths) for x in g]
            g_is = [x/iw_cum for x in g_is]
            g_d = [sum(x) for x in zip(g_is,g,s_g)]  
            stp = np.sum(f_update[n_sub](g[0],g[1],g[2],g[3],g[4],g[5],g[6]))
            print("step:",stp)
            if (stp/M<stp_snp/N or n_sub+1>= max_sub_iter):
                break
        n_sub_iter.append(n_sub)
        snap_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True)    
        
    print('Fine sessione ', k)
    rewards_subiter_data["rewardsSubIter"+str(k)]=rewards_sub_iter
    rewards_snapshot_data["rewardsSnapshot"+str(k)]= rewards_snapshot
    n_sub_iter_data["nSubIter"+str(k)]= n_sub_iter
    variance_sgd_data["variancceSgd"+str(k)] = variance_sgd
    variance_svrg_data["varianceSvrg"+str(k)]=variance_svrg
    importance_weights_data["importanceWeights"+str(k)] = importance_weights

    avg_return=np.array(avg_return)
    #plt.plot(avg_return)
    #plt.show()
rewards_subiter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_subiter_data.items() ]))
rewards_snapshot_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_snapshot_data.items() ]))
n_sub_iter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in n_sub_iter_data.items() ]))
variance_sgd_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_sgd_data.items() ]))
variance_svrg_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_svrg_data.items() ]))
importance_weights_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in importance_weights_data.items() ]))


rewards_subiter_data.to_csv("rewards_subiter_swimmer_vB.csv",index=False)
rewards_snapshot_data.to_csv("rewards_snapshot_swimmer_vB.csv",index=False)
n_sub_iter_data.to_csv("n_sub_iter_swimmer_vB.csv",index=False)
variance_sgd_data.to_csv("variance_sgd_vB_swimmer.csv",index=False)
variance_svrg_data.to_csv("variance_svrg_vB_swimmer.csv",index=False)
importance_weights_data.to_csv("importance_weights_vB_swimmer.csv",index=False)

