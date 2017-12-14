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
        x = unpack(f_train_SVRG_4v(observations[i],actions[i],d_rewards[i],iw))*np.sqrt(np.int(n_traj/n_part)/M)
        g = snap_grads[j]*np.sqrt(np.int(n_traj_s/n_part)/N)+x
        for s in range(i+1,i+np.int(n_traj/n_part)):
            iw = f_importance_weights(observations[s],actions[s])
            g_prov=unpack(f_train_SVRG_4v(observations[s],actions[s],d_rewards[s],iw))*np.sqrt(np.int(n_traj/n_part)/M)
            g+=snap_grads[j]*np.sqrt(np.int((n_traj_s)/n_part)/N) + g_prov
        g=g/n_traj*n_part
        i+=np.int(n_traj/n_part)
        j+=1
        svrg.append(g)
    return (np.diag(np.cov(np.matrix(svrg),rowvar=False)).sum())

    
    
load_policy=True
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(CartpoleEnv())
#env = GymEnv("InvertedPendulum-v1")
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
snap_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
back_up_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
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
T = 100
#We will collect M secondary trajectories
M = 10
#Number of sub-iterations
#m_itr = 100
# Number of iterations
#n_itr = np.int(10000/(m_itr*M+N))
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
learning_rate = 0.00005
#perc estimate
perc_est = 0.6
#tot trajectories
s_tot = 10000

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
eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

surr_on1 = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var*importance_weights_var)
surr_on2 = TT.sum(snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var)
grad_SVRG =[sum(x) for x in zip([eval_grad1, eval_grad2, eval_grad3, eval_grad4], theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
grad_SVRG_4v = [sum(x) for x in zip(theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
grad_var = theano.grad(surr_on1,params)

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

f_train_SVRG = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var, eval_grad1, eval_grad2, eval_grad3, eval_grad4,importance_weights_var],
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

alla = []
alla2 = []
alla3 = []
for k in range(10):
    alla4=[]
    if (load_policy):
        snap_policy.set_param_values(np.loadtxt('policy_novar.txt'), trainable=True)
        policy.set_param_values(np.loadtxt('policy_novar.txt'), trainable=True)
    avg_return = np.zeros(s_tot)
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    j=0
    while j<s_tot-N:
        paths = parallel_sampler.sample_paths_on_trajectories(snap_policy.get_param_values(),N,T,show_bar=False)
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
        
        b=compute_snap_batch(observations[0:porz],actions[0:porz],d_rewards[0:porz],porz,partition)
        f_update(s_g[0],s_g[1],s_g[2],s_g[3])
        avg_return[j-N:j] = np.repeat(np.mean([sum(p["rewards"]) for p in paths]),N)
    
        var_sgd = np.cov(np.matrix(b),rowvar=False)
        var_batch = (var_sgd)*(porz/partition)/M
        
        print(str(j-1)+' Average Return:', avg_return[j-1])
        
        back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
        n_sub = 0
        while j<s_tot-M:
            iw_var = f_importance_weights(observations[0],actions[0])
            s_g_is = var_SVRG(observations[0], actions[0], d_rewards[0],iw_var)
            s_g_fv_is = [unpack(s_g_is)]
            for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
                iw_var = f_importance_weights(ob, ac)
                s_g_is = var_SVRG(ob, ac, rw,iw_var)
                s_g_fv_is.append(unpack(s_g_is))
            var_svrg = (estimate_variance(observations[porz:],actions[porz:],d_rewards[porz:],b,N-porz,porz,partition,M,N))
            var_dif = var_svrg-(np.diag(var_batch).sum())
            alla2.append(var_svrg)
            alla3.append((np.diag(var_batch).sum()))
            alla4.append(np.mean(iw_var))
            #eigval = np.real(np.linalg.eig(var_dif)[0])
            if (var_dif>0 or np.mean(iw_var)<0.5):
                policy.set_param_values(back_up_policy.get_param_values(trainable=True), trainable=True) 
                break
            #print(np.sum(eigval))
            j += M
            n_sub+=1
            sub_paths = parallel_sampler.sample_paths_on_trajectories(snap_policy.get_param_values(),M,T,show_bar=False)
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
            iw = f_importance_weights(sub_observations[0],sub_actions[0])
            
            back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
            
            g = f_train_SVRG(sub_observations[0],sub_actions[0],sub_d_rewards[0],s_g[0],s_g[1],s_g[2],s_g[3],iw)
            for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
                iw = f_importance_weights(ob,ac)
                g = [sum(x) for x in zip(g,f_train_SVRG(ob,ac,rw,s_g[0],s_g[1],s_g[2],s_g[3],iw))]
            g = [x/len(sub_paths) for x in g]
            f_update(g[0],g[1],g[2],g[3])
            param_sup = snap_policy.get_param_values()
            sb = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),M,T,show_bar=False)
            snap_policy.set_param_values(param_sup)
            avg_return[j-M:j] = np.repeat(np.mean([sum(p["rewards"]) for p in sb]),M)
            #print(str(j)+' Average Return:', avg_return[j])
        snap_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True)    
        
    plt.plot(avg_return[::10])
    plt.show()
    plt.plot(alla2)
    plt.plot(alla3)
    plt.show()
    alla.append(avg_return)
alla_mean = [np.mean(x) for x in zip(*alla)]
plt.plot(alla_mean)
np.savetxt("GPOMDP_SVRG_5e-5_ada",alla_mean)

gpomdp = np.loadtxt("GPOMDP_l5e-05")
#gpomdp_svrg_ada_wb = np.loadtxt("GPOMDP_SVRG_5e-5_ada_wb")
#gpomdp_svrg=np.loadtxt("GPOMDP_SVRG_5e-5")
#gpomdp_svrg_ada_wb = np.loadtxt("GPOMDP_SVRG_5e-5_ada_wb")
gpomdp_svrg_ada_wb_bv_m7 = np.loadtxt("GPOMDP_SVRG_5e-5_ada_b2")
#gpomdp_svrg_ada_wb_bv_m5 = np.loadtxt("GPOMDP_SVRG_5e-5_ada_b2_m5")
#gpomdp_svrg_ada_wb_bv_m3 = np.loadtxt("GPOMDP_SVRG_5e-5_ada_b2_m3")
#gpomdp_svrg_ada_wb_bv_s15 = np.loadtxt("GPOMDP_SVRG_5e-5_ada_b2_s15")
#
plt.plot(gpomdp)
#plt.plot(gpomdp_svrg)
#plt.plot(gpomdp_svrg_ada_wb[::10])
plt.plot(gpomdp_svrg_ada_wb_bv_m7[::10])
#plt.plot(gpomdp_svrg_ada_wb_bv_m3[::10])
#plt.plot(gpomdp_svrg_ada_wb_bv_m5[::10])
#plt.plot(gpomdp_svrg_ada_wb_bv_s15[::10])
#plt.legend(['gpomdp','gpomdp_svrg','gpomdp_svrg_ada_wb','gpomdp_svrg_m7','gpomdp_svrg_s15'], loc='lower right')
#plt.savefig("adapt_nnv.jpg", figsize=(32, 24), dpi=160)
plt.show()
#uni = np.ones(640,dtype=np.int)
#for i in range(40):
#    uni[i*16]=10
#scal_svrg = np.repeat(gpondp_svrg,uni)
#plt.plot(gpondp)
#plt.plot(scal_svrg )
#plt.legend(['gpondp','gpondp_svrg'], loc='lower right')
#plt.savefig("gpondp_5e-6.jpg", figsize=(32, 24), dpi=160)