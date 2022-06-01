import matplotlib.pyplot as plt
from reinforcement_learning.tabular.q_learning import QLearning
import gym
import gym_codebending.envs.gregworld

env = gym.make('GregWorld-v0', delay=150)

obs_space = env.observation_space.n
action_space = env.action_space.n

learning_rate = 0.7500446225972569
discount_factor = 0.7028582419812996
epsilon = 0.0018246964376493488

n_episodes = 200
#plt.ion()

q_learning = QLearning((obs_space, action_space), learning_rate, discount_factor, epsilon)
total_avg_reward = 0
avg_rewards = []
for episode in range(n_episodes):
    obs = env.reset()
    avg_reward = 0
    step_count = 0
    while True:
        step_count += 1
        action = q_learning.get_action(obs)
        next_state, reward, done, info = env.step(action)
        q_learning.update(obs, action, reward, next_state)

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
        action = q_learning.get_best_action(obs)
        next_state, reward, done, info = env.step(action)
        env.render()
        obs = next_state
        if done:
            break

