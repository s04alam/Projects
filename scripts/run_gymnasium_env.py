import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.grid_world import GridWorldEnv
from scripts import rl_algorithms


import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import sys

print("Gym version:", gym.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

def visualize_values(V, title):
    """
    method to visualize the valuevector

    Args:
        V (list[int]): the valuevector
        title (string): the title for the plot

    Returns:
        None
    """
    size = int(sqrt(V.shape[0]))
    ax = sns.heatmap(V.reshape(size, size),
                 annot=True, square=True,
                 cbar=False, cmap='Blues',
                 xticklabels=False, yticklabels=False, )
    plt.title(title)
    plt.show()

def transform_q_table(qtable):
    """
        transform the Q table such that you have per state the 4 different q-values

        Args:
            qtable (list[list[float]]): shape (4, env.observation_space.n)

        Returns:
            q_per_state (list[list[float]]): shape (env.observation_space.n,4)
        """
    q_per_state = np.zeros((len(qtable[0]),len(qtable))) #shape (numstates, numactions)
    for action in range(len(qtable)):
        for state in range(len(qtable[action])):
            q_per_state[state][action] = qtable[action][state]
    return q_per_state

def visualize_policy(env, policy):
    env.reset(seed=SEED, render=False)
    for i in range(100):
        _,_, terminated, _,_ = env.follow_policy_stochastic(policy, render=True)
        if terminated:
            break

"data analysis"
def rmse_mc_pred(v_values, dp_value):
    """
    v_values: array of state-values, one for each episode
    dp_value: array of state-values (only one, as ground truth)
    """
    def helper(v_value, dp_value):
        rmse = np.sqrt(np.mean((np.subtract(v_value, dp_value))**2))
        return rmse

    rmse_array = np.array([helper(v_value, dp_value) for v_value in v_values])
    return rmse_array

# Create the environment
env = gym.make("GridWorld-v0", render_mode="human", size=5)
env = env.unwrapped

# Reset the environment
SEED = 42
np.random.seed(SEED)
state, _ = env.reset(seed=SEED,render=False) #you can change render=True to get a visualization
algos = rl_algorithms.RLAlgorithms(env)
action_map = {0: '→', 1: '↑', 2: '←', 3: '↓', 'X': 'X'}

"random policy"
random_policy = algos.random_policy()

"policy evaluation"
# value_vector = algos.policyEval()
# visualize_values(value_vector, title="Policy Evaluation with random policy")

"policy improvement --> seems to work"
# policy, currentValueVector = algos.policy_iteration()
# visualize_values(currentValueVector, title="Policy Iteration")
# policy_grid = np.array([action_map[a] for a in policy]).reshape(env.size,env.size)
# print("policy improvement:\n", policy_grid)
# visualize_policy(env,policy)



"value iteration"
# vi_policy, vi_vectors = algos.value_iteration()
# visualize_values(vi_vectors, title="Value Iteration ")
# policy_grid = np.array([action_map[a] for a in vi_policy]).reshape(env.size,env.size)
# print("Value iteration:\n", policy_grid)
# visualize_policy(env,vi_policy)



"first visis MC prediction  --> weird results, we have 0's. this is because with a random policy you can have states whihc are never visited, and therefore won't have a value. using this policy usually gives valus for all states"
# pol = [3,0,0,0,3,0,0,3,3,3,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0]
# V = algos.first_visit_MC_prediction(policy=pol,num_episodes=1000)
# visualize_values(V, title="First Visit MC Prediction")

"on-policy first visit MC control --> Q-values of state 15 are 1111, of holes -1-1-1-1 ???"
# Q,final_e_soft_policy = algos.first_visit_MC_control(num_episodes=100) #learned Q values and a probabiliy distribution for every state
# best_actions_dict = { state: np.argmax(actions) if not np.all(actions == np.zeros(env.action_space.n)) else 'X' for state, actions in Q.items() }
# #this list contains for every state the action with the highest q-value
# final_policy_lst =  [np.argmax(actions) for state, actions in Q.items() ]
# print("Learned Q-Values:", )
# for state,values in Q.items():
#     print(f"state {state} values {values}")
# policy_grid = np.array([action_map[a] for a in best_actions_dict.values()]).reshape(env.size,env.size)
# print(policy_grid)
# visualize_policy(env,final_policy_lst)

"TD prediction --> doubtful, lot's of 0's. once again, need to manually unsure enough exploration"
# pol = [3,0,0,0,3,0,0,3,3,3,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0]
# td_values, value_over_time = algos.temporaldiff_pred(episodes=100, current_policy=pol)
# visualize_values(td_values, title="Temporal Difference Prediction")

"sarsa --> works"
# sarsa_policy, qtable, q_values_over_time = algos.sarsa(num_episodes=1000)
# policy_grid = np.array([action_map[a] for a in sarsa_policy]).reshape(env.size,env.size)
# print(transform_q_table(qtable=qtable))
# print(policy_grid)
# visualize_policy(env,sarsa_policy)

"q-learning"
# qlearning_policy, qtable, q_values_over_time = algos.q_learning(num_episodes=1000)
# policy_grid = np.array([action_map[a] for a in qlearning_policy]).reshape(env.size,env.size)
# print(policy_grid)
# print("states and the q-values of all 4 actions\n", transform_q_table(qtable=qtable))
# visualize_policy(env,qlearning_policy)


"trying out some shit"
#Navigating in the env
# pol = [0, 3, 1, 1, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
# for i in range(100):
#     #state, reward, terminated, truncated, info = env.follow_deterministic_policy(pol, render=True)  # Move DOWN --> for some reason, step(1) is down, while it should be step(3) is down
#     print(f"with action {pol[state]} from state {state}. Reward of step:{reward} Done: {terminated}")
#     if terminated:
#         break

#random policy visualized
# for _ in range(20):
#     # You can use any action from the action space. For example, let's just choose 'right'.
#     action = env.action_space.sample()  # Sample a random action
#
#     # Take a step in the environment
#     observation, reward, terminated, truncated, info = env.step(action, render=True)
#
#     # Render the environment to update the display
#     env.render()
#
#     # If the episode is done, break the loop
#     if terminated:
#         break





# Close the environment window after the episode ends
env.close()



