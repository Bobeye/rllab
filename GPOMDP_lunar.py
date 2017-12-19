from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import sgd
from rllab.misc import ext
from rllab.envs.gym_env import GymEnv


import matplotlib.pyplot as plt

load_policy=False
# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(GymEnv("LunarLanderContinuous-v2"))
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
T = 1000
# Number of iterations
n_itr = 1000
# Set the discount factor for the problem
discount = 0.995
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
# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)

surr = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)

params = policy.get_params(trainable=True)

grad = theano.grad(surr, params)

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.matrix('eval_grad3',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

f_train = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)
f_update = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
    outputs = None,
    updates = sgd([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params, learning_rate=learning_rate)
)

alla = []
for i in range(10):
    if (load_policy):
        policy.set_param_values(np.loadtxt('policy_novar.txt'), trainable=True)        
    avg_return = np.zeros(n_itr)
    #np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
    for j in range(n_itr):
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
        s_g = f_train(observations[0], actions[0], d_rewards[0])
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            s_g = [sum(x) for x in zip(s_g,f_train(ob, ac, rw))]
        s_g = [x/len(paths) for x in s_g]
        
        f_update(s_g[0],s_g[1],s_g[2],s_g[3])
        avg_return[j] = np.mean([sum(p["rewards"]) for p in paths])
        if (j%10==0):
            print(str(j)+' Average Return:', avg_return[j])
            
    plt.plot(avg_return[::10])
    plt.show()
    alla.append(avg_return)
alla_mean = [np.mean(x) for x in zip(*alla)]
plt.plot(alla_mean)
np.savetxt("GPOMDP_asda",alla_mean)

obs = env.reset()
done=0
i=0
rcum= 0
while not done:
    ac = policy.get_action(obs)
    obs,rew,done,info = env.step(ac[0])
    rcum += rew
    env.render()
    i+=1
    
    
