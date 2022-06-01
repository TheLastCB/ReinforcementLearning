import matplotlib.pyplot as plt
from reinforcement_learning.tabular.sarsa import Sarsa
import gym
import gym_codebending.envs.gregworld
from tqdm import tqdm


env = gym.make('CartPolePyMunk-v0')

obs_space = env.observation_space_table
action_space = env.action_space.n

learning_rate = 0.10355477051468118
discount_factor = 0.9094660817865091
epsilon = 0.14775829391274092
MAX_POSSIBLE_STEPS = 1000
n_episodes = 5000
plt.ion()

sarsa = Sarsa(obs_space + [action_space], learning_rate, discount_factor, epsilon)

rewards = []
steps_count = []
for episode in tqdm(range(n_episodes)):
    obs = env.reset()
    avg_reward = 0
    steps = 0
    while True:
        steps += 1
        action = sarsa.get_action(obs)
        next_state, reward, done, info = env.step(action)
        next_action = sarsa.get_action(next_state)
        sarsa.update(obs, action, reward, next_state, next_action)

        avg_reward += reward

        obs = next_state
        if episode % 1000 == 0:
            env.render()
        if done:
            break

    rewards.append(avg_reward / steps)
    steps_count.append(steps / MAX_POSSIBLE_STEPS)
    plt.clf()
    plt.plot(rewards, color="red")
    plt.draw()
    plt.pause(0.01)


for episode in range(10):
    obs = env.reset()
    env.render()
    while True:
        action = sarsa.get_best_action(obs)
        next_state, reward, done, info = env.step(action)
        env.render()
        obs = next_state
        if done:
            break

