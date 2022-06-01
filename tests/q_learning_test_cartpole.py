import matplotlib.pyplot as plt
from reinforcement_learning.tabular.q_learning import QLearning
import gym
import gym_codebending.envs.gregworld
from tqdm import tqdm


env = gym.make('CartPolePyMunk-v0')

obs_space = env.observation_space_table
action_space = env.action_space.n

learning_rate = 0.07952899711841932
discount_factor = 0.833231506372483
epsilon = 0.12895946063789

n_episodes = 5000
plt.ion()

q_learning = QLearning(obs_space + [action_space], learning_rate, discount_factor, epsilon)
total_avg_reward = 0
avg_rewards = []
for episode in tqdm(range(n_episodes)):
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
        if episode % 1000 == 0:
            env.render()
        if done:
            break

    avg_reward /= step_count
    avg_rewards.append(avg_reward)
    plt.clf()
    plt.plot(avg_rewards, color="red")
    plt.draw()
    plt.pause(0.01)

    total_avg_reward += avg_reward

total_avg_reward /= n_episodes
print(total_avg_reward)
plt.plot(avg_rewards)
plt.show()

for episode in range(10):
    obs = env.reset()
    env.render()
    while True:
        action = q_learning.get_best_action(obs)
        next_state, reward, done, info = env.step(action)
        env.render()
        obs = next_state
        if done:
            break

