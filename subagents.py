"""Various specific Agents."""

from agent import Agent
from gym.spaces import Discrete
from utils import argmax

import copy
import random

class OneActionAgent(Agent):

	def __init__(self, id, env, total_reward = 0, policy_parameters=None):
		super(OneActionAgent, self).__init__(id, env, total_reward, policy_parameters)
		self.my_action = env.action_space.sample()
		
	def _policy(self, current_state):
		"""By default, the policy is to just select a random action."""
		
		return self.my_action

class FixedPolicyAgent(Agent):
	"""policy_parameters is a dictionary of states to actions."""
	
	def __init__(self, id, env, total_reward = 0, policy_parameters=None):
		super(FixedPolicyAgent, self).__init__(id, env, total_reward, policy_parameters)
		
		if self.policy_parameters == None:
			# Build a random policy on the fly.
			self.policy_parameters = {}
	
	def _policy(self, current_state):
		"""By default, the policy is to just select a random action."""

		if current_state not in self.policy_parameters:
			self.policy_parameters[current_state] = self.env.action_space.sample()
		
		return self.policy_parameters[current_state]
		
class QLearningAgent(Agent):
	"""policy_parameters is a list [eps, alpha, A, B, gamma]."""
	
	def __init__(self, id, env, total_reward = 0, policy_parameters=None):
		
		# For now we only support discrete action spaces and observation spaces
		assert type(env.action_space) == Discrete
		assert type(env.observation_space) == Discrete

		super(QLearningAgent, self).__init__(id, env, total_reward, policy_parameters)
		
		self.Q = dict([(s, dict([(a, 0) for a in range(env.action_space.n)])) for s in range(env.observation_space.n)])
		
		if self.policy_parameters == None:
			self.policy_parameters = [0.1, 0.1, 1, 1, 0.1]
		
		self.eps, self.alpha, self.A, self.B, self.gamma = self.policy_parameters
		self.initial_policy_parameters = copy.deepcopy(self.policy_parameters)
			
	def _policy(self, current_state):
		"""Select an action epsilon-greedily."""
		
		if random.uniform(0, 1) < self.eps:
			action = self.env.action_space.sample()
		else:
			action = argmax(range(self.env.action_space.n), lambda a : self.Q[self.current_state][a])
		
		return action
		
	def _update(self, action, reward):
		"""Update Q according to reward received."""
		
		Q = self.Q

		maxaction = argmax(range(self.env.action_space.n), lambda a : Q[self.current_state][a] - Q[self.previous_state][action])
		maxactiondiff = Q[self.current_state][maxaction] - Q[self.previous_state][action]
				
		Q[self.previous_state][action] += self.alpha*(self.A*reward + self.B*self.gamma*maxactiondiff)
