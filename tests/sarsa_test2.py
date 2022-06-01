import matplotlib.pyplot as plt
from reinforcement_learning.tabular.sarsa import Sarsa
import gym
import gym_codebending.envs.gregworld

env = gym.make('GregWorld-v0', delay=150)

obs_space = env.observation_space.n
action_space = env.action_space.n

learning_rate = 0.9902378561552334
discount_factor = 0.5588146116955788
epsilon = 0.0032454981949265518

n_episodes = 200
#plt.ion()

sarsa = Sarsa((obs_space, action_space), learning_rate, discount_factor, epsilon)
total_avg_reward = 0
avg_rewards = []
for episode in range(n_episodes):
    obs = env.reset()
    avg_reward = 0
    step_count = 0
    while True:
        step_count += 1
        action = sarsa.get_action(obs)
        next_state, reward, done, info = env.step(action)
        next_action = sarsa.get_action(next_state)
        sarsa.update(obs, action, reward, next_state, next_action)

        avg_reward += reward

        obs = next_state
        #env.render()
        if done:
            break

    avg_reward /= step_count
    avg_rewards.append(avg_reward)
    #plt.clf()
    #plt.plot(avg_rewards, color="red")
    #plt.draw()
    #plt.pause(0.01)

    total_avg_reward += avg_reward

total_avg_reward /= n_episodes


for episode in range(3):
    obs = env.reset()
    env.render()
    while True:
        action = sarsa.get_best_action(obs)
        next_state, reward, done, info = env.step(action)
        env.render()
        obs = next_state
        if done:
            break

