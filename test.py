import gym
env = gym.make('FrozenLake-v0')
print env.action_space
assert type(env.action_space) == gym.spaces.Discrete
print env.observation_space
# for i_episode in xrange(1):
#     observation = env.reset()
#     for t in xrange(1):
#         env.render()
#         print observation
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print "Episode finished after {} timesteps".format(t+1)
#             break
