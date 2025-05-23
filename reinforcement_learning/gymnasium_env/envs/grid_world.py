
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.observation_space.n = size**2
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        self.action_space.n = 4

        self.transition_probabilities = {
            0: [0.5, 0.25, 0, 0.25],
            1: [0.25, 0.5, 0.25, 0],
            2: [0, 0.25, 0.5, 0.25],
            3: [0.25, 0, 0.25, 0.5]
                }
        
        self.P = [] #defined in reset()

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
        }


        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_P(self):
        return self.P
    
    def _get_obs(self):
        return self.state_to_int(self._agent_location)
    
    def state_to_int(self, state):
        "state: np.array([horizontal,vertical])"
        return state[0] + self.size * state[1]

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None, render=False):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initiaze location of agent, goal and 'holes'
        self._agent_location = np.array([0,0]) #self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = np.array([self.size -1, self.size -1])
        #self._punish_location = np.array([[3, 0],[0, 3], [2,2], [3,3]])
        self._punish_location = np.array([[3, 3]])

        #defining P. We want to have P since P is used in the algorithms implemented for frozen lake
        for i in range(self.size):
            for j in range(self.size):
                state = np.array([i,j])
                l2 = []
                for prefered_action in range(self.action_space.n):
                    l1 = []
                    for actual_action in range(self.action_space.n):
                        #terminal state (either goal or hole)
                        if np.array_equal(state, self._target_location) or np.any(np.all(self._punish_location == state, axis=1)):
                            p = 0
                        #no-terminal state
                        else:
                            p = self.transition_probabilities[prefered_action][actual_action]
                        next, reward, terminated = self.hypothetical_step(actual_action,state)
                        tupl = (p,self.state_to_int(next),reward,terminated)
                        l1.append(tupl)
                    l2.append(l1)
                self.P.append(l2)

        state = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and render==True:
            self._render_frame()

        return state, info
    
    def follow_deterministic_policy(self,policy,render=True):
        """
        Let the agent take one deterministic step, specified by the policy

        Args:
            policy list(int): a list of length `self.observation_space.n` containing integers from self.action_space

        Returns:
            state (int): the new location after taking one deterministic step
            reward (float): the reward returned after taking one deterministic step
            terminated (boolean) : whether the episode has ended in a terminal state
            truncated (boolean): whether the episode is truncated
            info: float: distance between agent and target
        """
        state = self.state_to_int(self._agent_location)
        action = policy[state]
        state, reward, terminated, truncated, info = self.step(action, stochastic=False, render=render)
        return state, reward, terminated, truncated, info

    def follow_policy_stochastic(self,policy,render=True):
        """
        Let the agent take one step, specified by the policy, in a stochastic environment.

        Args:
            policy list(int): a list of length `self.observation_space.n` containing integers from self.action_space

        Returns:
            state (int): the new location after taking one deterministic step
            reward (float): the reward returned after taking one deterministic step
            terminated (boolean) : whether the episode has ended in a terminal state
            truncated (boolean): whether the episode is truncated
            info: float: distance between agent and target
        """
        state = self.state_to_int(self._agent_location)
        action = policy[state]
        state, reward, terminated, truncated, info = self.step(action, stochastic=True, render=render)
        return state, reward, terminated, truncated, info

    def step(self, prefered_action, stochastic=True, render=False):
        """
            Take a step in the environment.

            Args:
                prefered_action (int): the prefered action, a number from 0 to 3
                stochastic (bool): whether the environment is stochastic
                render (bool): Whether to render the environment.

            Returns:
                state (int): the new location after taking one deterministic step
                reward (float): the reward returned after taking one deterministic step
                terminated (boolean) : whether the episode has ended in a terminal state
                truncated (boolean): whether the episode is truncated
                info: float: distance between agent and target
            """

        def choose_next_action(prefered_action):
                """
                returns the actual action the agent will take given the action it chooses to take.

                Args:
                    prefered_action (int): the prefered action, a number from 0 to 3

                Returns:
                    next_action (int): the actual action the agent will take
                """
                actions = [0,1,2,3]
                probabilities = self.transition_probabilities[prefered_action]
                next_action = random.choices(actions, probabilities)[0]  # Randomly choose next action based on transition probabilities
                return next_action

        if stochastic:
            actual_action = choose_next_action(prefered_action)
        else:
            actual_action = prefered_action
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[actual_action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target or a punish state
        terminated = np.array_equal(self._agent_location, self._target_location) 
        is_punish_state =  np.any(np.all(self._punish_location == self._agent_location, axis=1))

        #define the rewards for reaching a goal/hole state or transitioning
        if is_punish_state:
            reward = -5
            terminated = True
        elif terminated:
            reward = 5
        else:
            reward = -0.1

        state = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and render==True:
            self._render_frame()

        return state, reward, terminated, False, info
    
    def hypothetical_step(self, actual_action, current_state):
        """
        Take a hypothetical step in the environment.The agent does not actually move. This is used to check the result of an action without actually executing it.

        Args:
            actual_action (int): the action the agent is going to take, a number from 0 to 3
            current_state (MultiDiscrete([size, size])): the current location of the agent

        Returns:
            next (int): the new location after taking one deterministic hypothetical step
            reward (float): the reward returned after taking this step
            terminated (boolean) : whether the episode has ended in a terminal state after taking this step
        """
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[actual_action]
        # We use `np.clip` to make sure we don't leave the grid
        next = np.clip(
            current_state + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(next, self._target_location) 
        is_punish_state =  np.any(np.all(self._punish_location == next, axis=1))

        if is_punish_state: 
            reward = -1
            terminated = True
        elif terminated:
            reward = 1
        else:
            reward = -0.1

        return next, reward, terminated
        
    


    "what follows are all visualization methods"

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for loc in self._punish_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * loc,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                100,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                100,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def visualize_P(self):
        def action_str(action_int):
            if action_int == 0:
                return "left"
            elif action_int == 1:
                return "down"
            elif action_int == 2:
                return "right"
            elif action_int == 3:
                return "up"
            else:
                return "invalid"

        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                print(f"state {state} action {action}")
                lst_for_action = self.P[state][action]
                for info in lst_for_action:
                    trans_prob, nxt_state, reward, isTerminal = info
                    print(f"\t\ttransition probability: {trans_prob} from {state} to {nxt_state}, which is terminal {isTerminal} and this step has reward {reward}") #state 0, acti
            
        return None 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
