from reinforcement_learning.tabular.sarsa import Sarsa
import gym
from gym_codebending.envs import gregworld
import numpy as np
from tqdm import tqdm
import optuna
from optuna.visualization import plot_intermediate_values, plot_param_importances

import pickle

optuna.logging.set_verbosity(optuna.logging.WARN)

env = gym.make('CartPolePyMunk-v0')

obs_space = env.observation_space_table
action_space = env.action_space.n


def logging_callback(current_study, frozen_trial):
    previous_best_value = current_study.user_attrs.get("previous_best_value", None)
    if previous_best_value != current_study.best_value:
        current_study.set_user_attr("previous_best_value", current_study.best_value)
        print(f"Trial {frozen_trial.number} finished with best value: {frozen_trial.value} and parameters: {frozen_trial.params}.")


def objective(trial: optuna.Trial):
    learning_rate = trial.suggest_float('learning_rate', 0, 1)
    discount_factor = trial.suggest_float('discount_factor', 0, 1)
    epsilon = trial.suggest_float('epsilon', 0, 1)

    n_episodes = trial.suggest_int('n_episodes', 1000, 5000, 1000)

    # n_episodes = 5000
    sarsa = Sarsa(obs_space + [action_space], learning_rate, discount_factor, epsilon)
    total_avg_reward = 0
    for episode in range(n_episodes):
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

            if done:
                break

        avg_reward /= steps
        total_avg_reward += avg_reward

    with open(f"sarsa_models/cartpole/{trial.number}.pickle", "wb") as fout:
        pickle.dump(sarsa, fout)

    total_avg_reward /= n_episodes
    return total_avg_reward


study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50, callbacks=[logging_callback], show_progress_bar=True)

'''
sarsa = pickle.load(open("sarsa_models/cartpole/98.pickle", 'rb'))

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

'''