from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from gym.envs.classic_control import CartPoleEnv
#from rlutils import Agent, enact_policy, evaluate_agent, state_cols
from collections import deque
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np



class Agent:
    """Base class for an agent. 
    
    Defines the operations needed to use and train the agent.
    """
    
    def get_action(self, state):
        """Generate an action given the state of the system"""
        raise NotImplementedError()
    
    def train(self, states):
        """Update an agent's policies given some examples of states"""
        raise NotImplementedError()


class DQNAgent(Agent):
    """An agent trained using DQN"""
    
    def __init__(self, size=128, gamma=0.95, epsilon=0.75, epsilon_decay=0.01, epsilon_min=0.1):
        """
        Args:
            gamma: Discount rate for future rewards
            epsilon: Exploration value
            epsilon_decay: How much we decay the rewards after each update
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_cols = [str(i) for i in range(size)]
        self.size = size
        # Make a model that predicts the value of a move and actions
        self.q_function = self.make_q_function()
        self.q_function.compile(loss='mse', optimizer='adam')
        
        # Memory for all observed moves
        self.memory = pd.DataFrame()
        self.max_memory = 2048
        
    def make_q_function(self):
        """Generate a Q-function that computes the value of both actions given state"""
        return Sequential([
            Dense(128, activation='relu', input_shape=(self.size,)),
            Dense(64, activation='relu', input_shape=(4,)),
            Dense(2, activation='linear')
        ])
    
    def get_action(self, state):       
        if np.random.random() < self.epsilon:
            # Choose action randomly
            return np.random.randint(0,4)
        else:
            # Compute the value of each move
            q_values = self.q_function.predict(state[np.newaxis, :])[0]
            # Pick the best value
            return np.argmax(q_values)
    
    def train(self, states):
        # Compute the next state for each state
        #  Numpy roll rotates the array from [1, ... N] to [2, ... N, 1]
        next_state_cols = []  # Stores the columns in the DataFrame that involve refitting the 
        for c in self.state_cols:
            next_state_cols.append(f'next_{c}')
            states[f'next_{c}'] = np.roll(states[c], -1)
        
        # Add new states to the memory
        self.memory = pd.concat([self.memory, states])
        
        # If needed, sample fewer points from the memory to keep it from becoming too big
        if len(self.memory) > self.max_memory:
            self.memory = self.memory.sample(self.max_memory)
        
        # Get compute the Q value for the next state
        #  The value is zero for the last point because there is no next state
        q_value_next = np.max(self.q_function.predict(self.memory[next_state_cols].values), axis=1)
        q_value_next = np.where(self.memory['done'], 0, q_value_next)
        
        # Compute the target Q-values
        q_target = self.memory['reward'].values + self.gamma * q_value_next
        
        # Save the old weights
        self.q_function.fit(self.memory[self.state_cols].values, q_target, shuffle=True, batch_size=32, verbose=False)
        
        # Last step, make the algorithm more greedy
        self.epsilon *= (1 - self.epsilon_decay)
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return 
    def enact_policy(self, env):
        """Run a policy on an environment given an agent"""
        states = []
        rewards = []
        actions = []
        dones = []

        # Reset the system
        state = env.reset()
        done = False

        # Step until "done" flag is thrown
        while not done:
            action = self.get_action(state)
            state, reward, done, data = env.step(action)  # Just push it to one side as an example
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)

        states = pd.DataFrame(states, columns=self.state_cols)
        states['reward'] = rewards
        states['step'] = np.arange(len(states))
        states['action'] = actions
        states['done'] = dones
        return states

    def evaluate_agent(self, env, n_episodes, train=True):
        """Evalaute an agent over many episodes of the cart-pole game

        Args:
            env: Test environment
            agent: Agent to use and train
            n_episodes: Number of episodes to run with the game
            train: Whether to train the agent after each episode
        Returns:
            Dataframe with the results of each episode
        """

        # Storage for the length of each episode
        length = []

        # Run the desired number of episodes
        for i in tqdm(range(n_episodes), leave=False):
            # Run the environment
            states = self.enact_policy(env)
            length.append(len(states))

            # Update agent, if desired
            if train:
                self.train(states)

        # Make the output
        return pd.DataFrame({'length': length, 'episode': np.arange(n_episodes)})

