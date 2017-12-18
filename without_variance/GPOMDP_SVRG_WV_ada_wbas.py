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

def compute_grad_snap(observations,actions,d_rewards,N,T):
        minT=T
        cum_num = []
        cum_den = []
        for ob,ac,rw in zip(observations,actions,d_rewards):
            if minT>len(ob):
                minT=len(ob)
            x=f_baseline_g(ob, ac)
            z = [y**2 for y in x]
            index2 = np.arange(len(rw))
            prov_der_num = [y[i]*rw[i] for i in index2 for y in z ]
            prov_der_den = [y[i] for i in index2 for y in z]
            cum_num.append(prov_der_num)
            cum_den.append(prov_der_den)
        mean_num = []
        mean_den = []
        baseline = []
        for i in range(minT):
            mean_num.append(cum_num[0][len(x)*i:len(x)*(i+1)])
            mean_den.append(cum_den[0][len(x)*i:len(x)*(i+1)])
        index = np.arange(len(mean_num[0]))
        for i in range(minT):
                for j in range(1,len(cum_den)):
                    mean_num[i] = [mean_num[i][pos] + cum_num[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
                    mean_den[i] = [mean_den[i][pos] + cum_den[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
        for i in range(minT):
            mean_num[i] = [mean_num[i][pos]/N for pos in index]
            mean_den[i] = [mean_den[i][pos]/N for pos in index]
            baseline.append([mean_num[i][pos]/mean_den[i][pos] for pos in index])
        zero_grad = [mean_den[0][pos]*0 for pos in index]
        for i in range(minT,T):
            baseline.append(zero_grad)
        cum = zero_grad
        cum_iter = zero_grad
        s_g = f_train(observations[0], actions[0])
        index2 = np.arange(len(d_rewards[0]))
        s_g = [y[i] for i in index2 for y in s_g]
        for i in range(len(observations[0])):
            R = [(d_rewards[0][i]-baseline[i][pos])*s_g[len(zero_grad)*i:len(zero_grad)*(i+1)][pos] for pos in index]
            cum = [R[pos]+cum[pos] for pos in index]
            cum_iter = [R[pos]+cum_iter[pos] for pos in index]
        s_g_fv = [unpack(cum_iter)]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = f_train(ob, ac)
            index2 = np.arange(len(rw))
            s_g = [y[i] for i in index2 for y in s_g]
            cum_iter = zero_grad
            for i in range(len(ob)):
                R = [(rw[i]-baseline[i][pos])*s_g[len(zero_grad)*i:len(zero_grad)*(i+1)][pos] for pos in index]
                cum = [R[pos]+cum[pos] for pos in index]
                cum_iter = [R[pos]+cum_iter[pos] for pos in index]
            s_g_fv.append(unpack(cum_iter))
        cum = [cum[pos]/N for pos in index]
        return cum,s_g_fv
    
def compute_grad_svrg(observations,actions,d_rewards,M,T,add_full,fg):
        minT=T
        cum_num = []
        cum_den = []
        for ob,ac,rw in zip(observations,actions,d_rewards):
            if minT>len(ob):
                minT=len(ob)
            x=f_baseline_g(ob, ac)
            index = np.arange(len(x))
            x_snap=f_baseline_g_snap(ob, ac)
            iw = f_importance_weights(ob,ac)
            index2 = np.arange(len(rw))
            x_iw = [y[i]*iw[i] for i in index2 for y in x ]
            x_snap_bv = [y[i] for i in index2 for y in x_snap]
            index3 = np.arange(len(x_snap_bv))
            x_dif = [x_iw[i]-x_snap_bv[i] for i in index3]
            z = [y**2 for y in x_dif]
            prov_der_num = [z[len(x)*i:len(x)*(i+1)][pos]*rw[i] for i in index2 for pos in index ]
            prov_der_den = z
            cum_num.append(prov_der_num)
            cum_den.append(prov_der_den)
        mean_num = []
        mean_den = []
        baseline = []
        for i in range(minT):
            mean_num.append(cum_num[0][len(x)*i:len(x)*(i+1)])
            mean_den.append(cum_den[0][len(x)*i:len(x)*(i+1)])
        for i in range(minT):
                for j in range(1,len(cum_den)):
                    mean_num[i] = [mean_num[i][pos] + cum_num[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
                    mean_den[i] = [mean_den[i][pos] + cum_den[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
        for i in range(minT):
            mean_num[i] = [mean_num[i][pos]/M for pos in index]
            mean_den[i] = [mean_den[i][pos]/M+1e-16 for pos in index]
            baseline.append([mean_num[i][pos]/mean_den[i][pos] for pos in index])
        zero_grad = [mean_den[0][pos]*0 for pos in index]
        for i in range(minT,T):
            baseline.append(zero_grad)
        cum = zero_grad
        s_g = f_baseline_g(observations[0], actions[0])
        s_g_snap_p=f_baseline_g_snap(observations[0], actions[0])
        iw = f_importance_weights(observations[0], actions[0])
        index2 = np.arange(len(d_rewards[0]))
        s_g_iw = [y[i]*iw[i] for i in index2 for y in s_g ]
        s_g_snap = [y[i] for i in index2 for y in s_g_snap_p]
        for i in range(len(observations[0])):
            R = [(d_rewards[0][i]-baseline[i][pos])*(s_g_iw[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]-s_g_snap[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]) for pos in index]
            cum = [R[pos]+cum[pos] for pos in index]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = f_baseline_g(ob, ac)
            s_g_snap=f_baseline_g_snap(ob, ac)
            iw = f_importance_weights(ob, ac)
            index2 = np.arange(len(rw))
            s_g_iw = [y[i]*iw[i] for i in index2 for y in s_g ]
            s_g_snap = [y[i] for i in index2 for y in s_g_snap]
            for i in range(len(ob)):
                R = [(rw[i]-baseline[i][pos])*(-s_g_iw[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]+s_g_snap[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]) for pos in index]
                cum = [R[pos]+cum[pos] for pos in index]
        cum = [cum[pos]/M for pos in index]
        if (add_full):
            cum = [cum[pos] + fg[pos] for pos in index]
        return cum

def compute_var_svrg(observations,actions,d_rewards,M,T):
        minT=T
        cum_num = []
        cum_den = []
        for ob,ac,rw in zip(observations,actions,d_rewards):
            if minT>len(ob):
                minT=len(ob)
            x=f_baseline_g(ob, ac)
            index = np.arange(len(x))
            x_snap=f_baseline_g_snap(ob, ac)
            iw = f_importance_weights(ob,ac)
            index2 = np.arange(len(rw))
            x_iw = [y[i]*iw[i] for i in index2 for y in x ]
            x_snap_bv = [y[i] for i in index2 for y in x_snap]
            index3 = np.arange(len(x_snap_bv))
            x_dif = [x_iw[i]-x_snap_bv[i] for i in index3]
            z = [y**2 for y in x_dif]
            prov_der_num = [z[len(x)*i:len(x)*(i+1)][pos]*rw[i] for i in index2 for pos in index ]
            prov_der_den = z
            cum_num.append(prov_der_num)
            cum_den.append(prov_der_den)
        mean_num = []
        mean_den = []
        baseline = []
        for i in range(minT):
            mean_num.append(cum_num[0][len(x)*i:len(x)*(i+1)])
            mean_den.append(cum_den[0][len(x)*i:len(x)*(i+1)])
        for i in range(minT):
                for j in range(1,len(cum_den)):
                    mean_num[i] = [mean_num[i][pos] + cum_num[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
                    mean_den[i] = [mean_den[i][pos] + cum_den[j][len(x)*i:len(x)*(i+1)][pos] for pos in index]
        for i in range(minT):
            mean_num[i] = [mean_num[i][pos]/M for pos in index]
            mean_den[i] = [mean_den[i][pos]/M+1e-16 for pos in index]
            baseline.append([mean_num[i][pos]/mean_den[i][pos] for pos in index])
        zero_grad = [mean_den[0][pos]*0 for pos in index]
        for i in range(minT,T):
            baseline.append(zero_grad)
        cum = zero_grad
        s_g = f_baseline_g(observations[0], actions[0])
        iw = f_importance_weights(observations[0], actions[0])
        index2 = np.arange(len(d_rewards[0]))
        s_g_iw = [y[i]*iw[i] for i in index2 for y in s_g ]
        for i in range(len(observations[0])):
            R = [(d_rewards[0][i]-baseline[i][pos])*(s_g_iw[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]) for pos in index]
            cum = [R[pos]+cum[pos] for pos in index]
        s_g_fv = [unpack(cum)]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = f_baseline_g(ob, ac)
            iw = f_importance_weights(ob, ac)
            index2 = np.arange(len(rw))
            cum = zero_grad
            s_g_iw = [y[i]*iw[i] for i in index2 for y in s_g ]
            for i in range(len(ob)):
                R = [(rw[i]-baseline[i][pos])*(-s_g_iw[len(zero_grad)*i:len(zero_grad)*(i+1)][pos]) for pos in index]
                cum = [R[pos]+cum[pos] for pos in index]
            s_g_fv.append(unpack(cum))
        return s_g_fv 
    
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
bl = TT.vector()

# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)
snap_dist_info_vars = snap_policy.dist_info_sym(observations_var)

surr = - dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars)

params = policy.get_params(trainable=True)
snap_params = snap_policy.get_params(trainable=True)

importance_weights = dist.likelihood_ratio_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars,dist_info_vars)

grad = TT.jacobian(surr, params)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

surr_on1 = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var*importance_weights_var)
surr_on2 = TT.sum(snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var)
grad_SVRG =[sum(x) for x in zip([eval_grad1, eval_grad2, eval_grad3, eval_grad4], theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
grad_SVRG_4v = [sum(x) for x in zip(theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
grad_var = theano.grad(surr_on1,params)

cum_likelihood = dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars)
cum_likelihood_snap = dist.log_likelihood_sym_1traj_GPOMDP(actions_var, snap_dist_info_vars)

all_der, update_scan = theano.scan(lambda i, cum_likelihood: theano.grad(cum_likelihood[i], params), 
                                    sequences=TT.arange(cum_likelihood.shape[0]), 
                                    non_sequences=cum_likelihood)

all_der_snap, update_scan = theano.scan(lambda i, cum_likelihood_snap: theano.grad(cum_likelihood_snap[i], snap_params), 
                                    sequences=TT.arange(cum_likelihood_snap.shape[0]), 
                                    non_sequences=cum_likelihood_snap)

f_train = theano.function(
    inputs = [observations_var, actions_var],
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

f_baseline_g = theano.function(
    inputs = [observations_var, actions_var],
    outputs = all_der
)

f_baseline_g_snap = theano.function(
    inputs = [observations_var, actions_var],
    outputs = all_der_snap
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
        s_g,s_g_fv = compute_grad_snap(observations,actions,d_rewards,N,T)      
        f_update(s_g[0],s_g[1],s_g[2],s_g[3])
        avg_return[j-N:j] = np.repeat(np.mean([sum(p["rewards"]) for p in paths]),N)
    
        var_sgd = np.cov(s_g_fv,rowvar=False)
        var_batch = var_sgd/(M)
        var_fg = var_sgd/(N)
        
        print(str(j-1)+' Average Return:', avg_return[j-1])
        
        back_up_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
        n_sub = 0
        while j<s_tot-M:
            s_g_fv_is = compute_var_svrg(observations,actions,d_rewards,M,T)
            var_is = np.cov(s_g_fv_is,rowvar=False)
            m_is = np.mean(s_g_fv_is,axis=0)
            m_sgd = np.mean(s_g_fv,axis=0)
            cov= np.outer(s_g_fv_is[0]-m_is,s_g_fv[0]-m_sgd)
            for i in range(N-1):
              cov += np.outer(s_g_fv_is[i+1]-m_is,s_g_fv[i+1]-m_sgd)  
            for i in range(N):
              cov += np.outer(s_g_fv[i]-m_sgd,s_g_fv_is[i]-m_is)  
            cov = cov/N
            var_svrg = var_fg + var_is + var_batch - cov
            var_dif = var_svrg-var_batch
            #eigval = np.real(np.linalg.eig(var_dif)[0])
            if (np.trace(var_dif)>0):
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
            
            g = compute_grad_svrg(observations,actions,d_rewards,M,T,True,s_g)
            f_update(g[0],g[1],g[2],g[3])
            avg_return[j-M:j] = np.repeat(np.mean([sum(p["rewards"]) for p in sub_paths]),M)
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
np.savetxt("GPOMDP_SVRG_wbas",alla_mean)

gpomdp = np.loadtxt("GPOMDP_l5e-05")
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