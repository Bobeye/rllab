from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import sgd
from rllab.misc import ext


import matplotlib.pyplot as plt

load_policy=True
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(CartpoleEnv())
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False)
parallel_sampler.populate_task(env, policy)

# policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
# distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
# the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
# rllab.distributions.DiagonalGaussian
dist = policy.distribution
# We will collect 100 trajectories per iteration
N = 10
# Each trajectory will have at most 100 time steps
T = 100
# Number of iterations
n_itr = 1000
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
learning_rate = 0.00005

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
bl = TT.vector()
# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)

surr = - dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars)

params = policy.get_params(trainable=True)

grad = TT.jacobian(surr, params)

cum_likelihood = dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars)

all_der, update_scan = theano.scan(lambda i, cum_likelihood: theano.grad(cum_likelihood[i], params), 
                                    sequences=TT.arange(cum_likelihood.shape[0]), 
                                    non_sequences=cum_likelihood)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

f_train = theano.function(
    inputs = [observations_var, actions_var],
    outputs = grad
)
f_update = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
    outputs = None,
    updates = sgd([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params, learning_rate=learning_rate)
)

f_baseline_g = theano.function(
    inputs = [observations_var, actions_var],
    outputs = all_der
)
alla = []
for est in range(10):
    if (load_policy):
        policy.set_param_values(np.loadtxt('policy_novar.txt'), trainable=True)        
    avg_return = np.zeros(n_itr)
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    for j_int in range(n_itr):
        paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),N,T,show_bar=False)
        #baseline.fit(paths)
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
        s_g = f_train(observations[0], actions[0])
        index2 = np.arange(len(d_rewards[0]))
        s_g = [y[i] for i in index2 for y in s_g]
        for i in range(len(observations[0])):
            R = [(d_rewards[0][i]-baseline[i][pos])*s_g[len(zero_grad)*i:len(zero_grad)*(i+1)][pos] for pos in index]
            cum = [R[pos]+cum[pos] for pos in index]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = f_train(ob, ac)
            index2 = np.arange(len(rw))
            s_g = [y[i] for i in index2 for y in s_g]
            for i in range(len(ob)):
                R = [(rw[i]-baseline[i][pos])*s_g[len(zero_grad)*i:len(zero_grad)*(i+1)][pos] for pos in index]
                cum = [R[pos]+cum[pos] for pos in index]
        cum = [cum[pos]/N for pos in index]
        s_g = cum
        f_update(s_g[0],s_g[1],s_g[2],s_g[3])
        avg_return[j_int] = np.mean([sum(p["rewards"]) for p in paths])
        if (j_int%10==0):
            print(str(j_int)+' Average Return:', avg_return[j_int])
            
    plt.plot(avg_return[::10])
    plt.show()
    alla.append(avg_return)
alla_mean = [np.mean(x) for x in zip(*alla)]
plt.plot(alla_mean)
np.savetxt("GPOMDP_with_base",alla_mean)
