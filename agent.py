"""Superclass for RL agents."""

class Agent(object):
	"""Agents have a state policy, updates, and total accumulated reward.
	
	Policies are functions from states to actions. Default is random.
	
	"""
	def __init__(self, id, env, total_reward = 0, policy_parameters=None):
		self.id = id
		self.env = env
		self.total_reward = total_reward
		self.policy_parameters = policy_parameters
		self.current_state = env.reset()
		self.previous_state = env.reset()
			
	def _policy(self, current_state):
		"""By default, the policy is to just select a random action."""
		
		return self.env.action_space.sample()
		
	def _update(self, action=None, reward=0):
		"""Only policy parameters may be updated."""
		
		return
		
	def _takeStep(self, learning=True):
		next_action = self._policy(self.current_state)
		self.previous_state = self.current_state
		self.current_state, reward, done, info = self.env.step(next_action)

		if learning:
			self._update(next_action, reward)
		
		self.total_reward += reward

		return not done
		
	def reset(self, total_reward=0):
		self.env.reset()
		self.total_reward = total_reward
		
	def evaluate(self, lifespan = 100, learning=True, verbose=False):
		self.env.reset()
		for e in range(lifespan):
			if self._takeStep(learning):
				if verbose:
					print self.env.render()
			else:
				return self.total_reward
					
		return self.total_reward
