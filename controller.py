import gym
import subagents

TRIALS = 1000
EPISODES = 100
LIFESPAN = 100

agents = {'One action agent': subagents.OneActionAgent,
		  'Random action agent': subagents.Agent,
		  'Fixed policy agent': subagents.FixedPolicyAgent,
		  'Q learning agent': subagents.QLearningAgent
		 }

total_rewards = dict(zip(agents.keys(), [0.0 for _ in range(len(agents))]))

for _ in range(TRIALS):
	env = gym.make('FrozenLake-v0')

	for agent_id, agent_class in agents.items():
		agent = agent_class(agent_id, env)
		for _ in range(EPISODES):
			total_rewards[agent_id] += agent.evaluate(LIFESPAN)
			agent.reset()

# Average
for agent_id in total_rewards:
	print agent_id,
	print total_rewards[agent_id]/(TRIALS*EPISODES)
