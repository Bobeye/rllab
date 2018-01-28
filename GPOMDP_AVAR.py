from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import sgd
from lasagne.updates import adam
from rllab.misc import ext
import pandas as pd

load_policy=True
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(CartpoleEnv())
# Initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=True)
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
learning_rate = 0.001

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
# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)

surr = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)

params = policy.get_params(trainable=True)

grad = theano.grad(surr, params)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)
eval_grad5 = TT.vector('eval_grad4',dtype=grad[4].dtype)

f_train = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)
f_update = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4,eval_grad5],
    outputs = None,
    updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5], params, learning_rate=learning_rate)
)
runs_rewards = {}
all_policy_param_data = {}
parallel_sampler.initialize(4)
for k in range(10):
    if (load_policy):
#        policy.set_param_values(np.loadtxt('policy.txt'), trainable=True)
        policy.set_param_values(np.loadtxt('pc' + np.str(k+1) + '.txt'), trainable=True)
    avg_return = np.zeros(n_itr)
    rewards=[]
    all_policy_param = []
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    for j in range(n_itr):
        if (j%10==0):
                all_policy_param.append(policy.get_param_values())
        paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),N,T,show_bar=False)
        paths = paths[:N]
        observations = [p["observations"] for p in paths]
        actions = [p["actions"] for p in paths]
        d_rewards = [p["rewards"] for p in paths]
        rewards.append(np.array([sum(p["rewards"])for p in paths]))
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
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = [sum(x) for x in zip(s_g,f_train(ob, ac, rw))]
        s_g = [x/len(paths) for x in s_g]
        
        f_update(s_g[0],s_g[1],s_g[2],s_g[3],s_g[4])
        avg_return[j] = np.mean([sum(p["rewards"]) for p in paths])
        if(j%10==0):
            print(str(j)+' Average Return:', avg_return[j])
    runs_rewards["trajReturn"+str(k)]=rewards
    all_policy_param_data["policyParams"+str(k)] = all_policy_param
runs_rewards = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in runs_rewards.items() ]))
all_policy_param_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_policy_param_data.items() ]))
runs_rewards.to_csv("GPOMDP_AVAR_rewards100.csv",index=False)
all_policy_param_data.to_csv("param_policy_GPOMDP100.csv",index=False)