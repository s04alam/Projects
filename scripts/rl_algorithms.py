import numpy as np 
import random

class RLAlgorithms:

    def __init__(self, env):
        self.env = env
        self.random_p= self.random_policy()



    def random_policy(self):
        """
        returns a random policy

        Returns:
            random_policy (list[int]): the random policy, which is a list of lenght env.observation_space.n and contains integers in the range of env.action_space.n
        """
        #random policy list
        random_policy = []
        for state in range(self.env.observation_space.n):
            random_policy.append(self.env.action_space.sample())
        return random_policy
    
    # policy improvement based on random policy
    def policyEval(self, current_policy=None, gamma=0.9, theta=10**(-6)):
        """
        Implements policy evaluation.

        Args:
            current_policy (List[Int])
            gamma (Float)
            theta (Float)

        Returns:
            valuevector (np.array[float]): an array containing the value estimates for all the states
        """
        if current_policy == None:  
            current_policy=self.random_p
        valueVector_old = np.zeros(self.env.observation_space.n)
        valueVector_new = valueVector_old.copy() # updated for each iteration

        # the value of each state converges after a certain number of iterations

        def returnValue(self, valueVector, i, action):
            """
            valueVector: List[Float]
            i: Int
            action: Int (from policy)
            function which implements Bellman expectation equations as an iterative update rule
            returns the updated value for state i
            """
            expectation = 0
            for transitions in self.env.P[i][action]:
                trans_prob, next_state, reward, isTerminal = transitions
                expectation += trans_prob * (reward + gamma * valueVector[next_state])
            return expectation

        while True:
            for i in range(len(valueVector_old)):
                valueVector_new[i] = returnValue(self,valueVector_old, i, current_policy[i])

            if np.max(abs(valueVector_old - valueVector_new)) < theta:
                break
            else:
                valueVector_old = valueVector_new.copy()

        return valueVector_new
    
    #Policy Improvement --> this is policy iteration right??
    def policy_iteration(self, gamma=0.9):
        """
        gamma: Float
        policy: List[Int]

        Implements policy iteration, i.e. iteratively evaluating a policy and updating it.
        Returns:
        policy: List[Int]
        currentValueVector: List[Float]
        """
        policy = self.random_p
        currentValueVector = self.policyEval()

        while True:
            policy_stable = True
            proposed_policy = policy.copy()

            for state in range(self.env.observation_space.n):
                old_action = policy[state] #

                values_all_actions = []
                for action in self.env.P[state]:
                    expectation = 0
                    for p, next, reward, isTerminal in action:
                        expectation += p * (reward + gamma * currentValueVector[next])
                    values_all_actions.append(expectation)

                new_action = np.argmax(values_all_actions)
                proposed_policy[state] = new_action

                if old_action != new_action:
                    policy_stable = False

            if policy_stable:
                break

            policy = proposed_policy.copy() # otherwise no change in next iteration
            currentValueVector = self.policyEval(policy)

        return policy, currentValueVector
    
    # Value Iteration
    def value_iteration(self, gamma=0.9, theta=10**(-6)):
        """
        Param:
        gamma: Float
        theta: Float

        returns:
        policy_old: List[Int]
        valueVector_old: List[Float]
        """
        policy_old = self.random_p
        valueVector_old = np.zeros(self.env.observation_space.n)

        def expect_action(valueVector_old, state, gamma):
            """
            function which implements the Bellman equation
            returns the updated value for state i and the best action
            """
            max_expectation = -1
            best_action = -1
            a = 0

            for action in self.env.P[state]: #self.env.P[state] gives 4 lists: one for each action. One such a list contains 4 tuples.                
                expectation = 0
                for p, next, reward, isTerminal in action:
                # computes the expected reward given each action in the state
                    expectation += p * (reward + gamma*valueVector_old[next])
                if expectation > max_expectation:
                # update the best action if the expectated reward icreases
                    max_expectation = expectation
                    best_action = a
                a +=1
            return max_expectation, best_action

        while True:
        # one iteration of value and policy updating
            valueVector_new = valueVector_old.copy()
            policy_new = policy_old.copy()
            for i in range(len(valueVector_old)):
                # update the values and policy given the best action
                valueVector_new[i],policy_new[i] = expect_action(valueVector_old,i,gamma)
            # stop condition
            if np.max(abs(valueVector_old-valueVector_new)) < theta:
                break
            else:
                valueVector_old = valueVector_new.copy()
                policy_old = policy_new.copy()

        return policy_old, valueVector_old


    def first_visit_MC_prediction(self,  policy, gamma=0.9, num_episodes=10000):
        """
        Param:
        policy: List[Int]
        gamma: Float
        num_episodes: Int

        returns:
        V: List[Float]
        V_over_time: List[List[Float]]
        rewards_and_ep_over_time: List[List[int, int]]
        """
        # Initialize
        V = np.zeros(self.env.observation_space.n)  # List of
        returns = {s: [] for s in range(self.env.observation_space.n)}  # Dictionary {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: []}

        def generate_episode(policy):
            episode = []
            state, _ = self.env.reset()
            while True:
                action = policy[state]
                #state after taking action, reward of taking action, done: is next state terminal?
                next_state, reward, done, truncated, _ = self.env.step(action,render=False)
                episode.append((state, reward))
                state = next_state
                if done or truncated:
                    break
            return episode

        def G(episode, t, gamma):
            #base case, t = T
            if t == len(episode)-1:
                return episode[t][1]
            else:
                return episode[t][1] + gamma * G(episode, t+1, gamma)


        for episode in range(num_episodes):
            # Generate an episode
            episode = generate_episode(policy)
            # For each episode loop over steps
            G_val = 0
            for t in reversed(range(len(episode))):
                state, reward = episode[t]
                #G = gamma * G + reward
                G_val = G(episode, t, gamma)
                if state not in [x[0] for x in episode[:t]]:
                    returns[state].append(G_val)
                    V[state] = np.mean(returns[state])
        return V




    def first_visit_MC_control(self, epsilon = 0.5 , gamma=0.9, num_episodes=10000):
        """
        Param:
        epsilon_soft_policy: function
        gamma: Float
        epsilon: Float
        num_episodes: Int

        returns:
        policy: List[Int]
        Q: dict key: state, value: list of q-action values (probability distr)
        value_array: List[Float]
        q_value_over_time: List[dict] -> dict = key: state, value: list of q-action values (probability distr)
        rewards_and_ep_over_time: List[List[int, int]]
        """

        def epsilon_soft_policy(state, epsilon, A_star):
            """implements an epsilon soft policy
            returns both an action based on the epsilon soft policy and the epsilon soft policy itself (i.e. a probability distribution)
            """ 
            num_actions = self.env.action_space.n
            probabilities = np.full(num_actions, epsilon / num_actions)
            probabilities[A_star] = 1 - epsilon + (epsilon / num_actions)
            # Sample an action based on the probability distribution
            action = np.random.choice(np.arange(num_actions), p=probabilities)
            return (action, probabilities)
        
        def generate_episode(epsilon_soft_policy_func, epsilon: float, A_star: int):
            """
            returns a list of tuples (state, action, reward)
            """
            episode = []
            state, _ = self.env.reset(render=False)
            #state =  self.env.state_to_int(obs['agent'])
            while True:
                action = epsilon_soft_policy_func(state, epsilon, A_star)[0]
                next_state, reward, done, truncated, _ = self.env.step(prefered_action=action, render=False)
                episode.append((state, action, reward))
                state = next_state
                if done or truncated:
                    break
            return episode

        def G(episode, t, gamma):
            #base case, t = T
            if t == len(episode)-1:
                return episode[t][1]
            else:
                return episode[t][1] + gamma * G(episode, t+1, gamma)
        

        # Initialization
        #the following 3 lines are needed to initialize returns, Q and policy
        states = [state for state in range(self.env.observation_space.n)]
        actions = [action for action in range(self.env.action_space.n)]
        state_action_pairs = [(state, action) for state in states for action in actions]
        #returns
        returns = {pair: [] for pair in state_action_pairs}
        #Q(s,a) has form {state: [0 , 0 , 0 , 0]}
        Q = {state: np.zeros(self.env.action_space.n) for state in states}
        #A* (np.random....) is needed to initialize an arbitrary intial epsilon soft policy
        policy = [ epsilon_soft_policy(state, epsilon, np.random.choice(self.env.action_space.n))[1] for state in states]
        A_star = np.random.choice(self.env.action_space.n)

        

        for i in range(num_episodes):
            # Generate an episode
            episode = generate_episode(epsilon_soft_policy,epsilon,A_star)

            # For each episode loop over steps
            G_val = 0
            for t in reversed(range(len(episode) )):
                state, action, reward = episode[t]
                G_val = G(episode, t, gamma)
                #create an list of state action pairs that have occured in the episode to check for occurence
                state_action_lst = [x[0:2] for x in episode[:t]]
                #if the state,action pair has not appeared in the sequence yet
                if (state,action) not in state_action_lst:
                    returns[(state,action)].append(G_val) #append G to returns
                    Q[state][action] = np.mean(returns[(state,action)]) #the value of the pair (state,action) is updated
                    #finding the optimal action in St
                    A_star = np.argmax(Q[state])
                    policy[state] = epsilon_soft_policy(state, epsilon, A_star)[1]
        return Q,policy

    def temporaldiff_pred(self, gamma=0.9, alpha=0.1, current_policy=None, episodes=10000):

        valueVector = np.zeros(self.env.observation_space.n) # step-wise improvement
        value_over_time = []
        if current_policy == None:  
            current_policy=self.random_p

        for episode in range(episodes): # idk stopping condition, bc od step-wise improvement
            # one episode
            state, _ = self.env.reset()
            while True:
                action = current_policy[state]
                nxt_state, reward, isTerminal, done, info = self.env.step(action) #gives for example {'agent': array([0, 1]), 'target': array([3, 3])} -0.1 False False {'distance': 5.0}
                if done or isTerminal:
                    break
                else:
                    valueVector[state] += alpha * (reward + gamma * valueVector[nxt_state] - valueVector[state])
                    state = nxt_state
                
            value_over_time.append(valueVector.copy())

        return valueVector, value_over_time

    #Sarsa (on-policy TD control).
    def sarsa(self, gamma=0.9, alpha=0.1, epsilon=0.5, num_episodes=10000):
        qtable = np.zeros((self.env.action_space.n, self.env.observation_space.n))
        q_values_over_time = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            # state =  self.env.state_to_int(obs['agent'])
            done = False

            # choose policy from qtable (first iteration)
            choice = [0,1,2,3]
            choice.remove(np.argmax(qtable[:, state]))
            prob_greedy = 1 - epsilon + (epsilon / self.env.action_space.n)
            action = np.argmax(qtable[:, state]) if np.random.uniform(low=0, high=1) < prob_greedy else np.random.choice(choice)

            while not done:
                next_state, reward, done_s, truncated, info = self.env.step(action)
                #next_state = self.env.state_to_int(obs['agent'])
                done = done_s or truncated

                # choose action for Q(s', a')
                next_max = np.argmax(qtable[:, next_state])
                next_choice = [0,1,2,3]
                next_choice.remove(next_max)

                prob_greedy = 1 - epsilon + (epsilon / self.env.action_space.n)
                action_next = next_max if np.random.uniform(0, 1) < prob_greedy else np.random.choice(next_choice)
                qtable[action, state] += alpha * (reward + gamma * qtable[action_next, next_state] - qtable[action, state])
                state = next_state
                action = action_next
            # used for rmse
            q_values_over_time.append(qtable.copy())

        # finalize policy
        policy = np.argmax(qtable, axis=0)
        return policy, qtable, q_values_over_time


    #Q-learning (on-policy TD control). <- shagota, we can use Inas version, but for me it does not work such that I implemented one myself
    def q_learning(self, gamma=0.9, alpha=0.1, epsilon=0.5, num_episodes=10000):
        qtable = np.zeros((self.env.action_space.n, self.env.observation_space.n))
        q_values_over_time = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            #state =  self.env.state_to_int(obs['agent'])
            done = False

            while not done:
                # choose policy from qtable (first iteration)
                choice = [0,1,2,3]
                choice.remove(np.argmax(qtable[:, state]))
                prob_greedy = 1 - epsilon + (epsilon / self.env.action_space.n)
                action = np.argmax(qtable[:, state]) if np.random.uniform(low=0, high=1) < prob_greedy else np.random.choice(choice)

                next_state, reward, done_s, truncated, info = self.env.step(action)
                #next_state = self.env.state_to_int(obs['agent'])
                done = done_s or truncated

                qtable[action, state] += alpha * (reward + gamma * max(qtable[:, next_state]) - qtable[action, state])
                state = next_state

            # used for rmse
            q_values_over_time.append(qtable.copy())

        # finalize policy
        policy = np.argmax(qtable, axis=0)
        return policy, qtable, q_values_over_time





