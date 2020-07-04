import sys
import pandas as pd
from agent import DDGP
from quad_env1 import QuadRotorEnv
import csv
import matplotlib.pyplot as plt
import numpy as np

num_episodes = 500
task = QuadRotorEnv([10,20,30])
agent = DDGP(task)
best_score = -1000
best_x = 0
best_y = 0
best_z = 0
data = {}
reward_log = "reward.txt"
reward_labels = ['episode', 'reward']
reward_results = {x : [] for x in reward_labels}
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()
    score = 0

    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        if score > best_score:
            best_x = task._get_obs()[0][0]
            best_y = task._get_obs()[0][1]
            best_z = task._get_obs()[0][2]
        best_score = max(score, best_score)
        data[i_episode] = {'Episode': i_episode, 'Reward':score,'Action':action,'Best_Score':best_score,
                            'Position_x':task._get_obs()[0],'Position_y':task._get_obs()[1],'Position_z':task._get_obs()[2]}
        if done:
            print("\rEpisode = {:4f}, score = {:7.3f} (best = {:7.3f}), last_position = ({:5.1f},{:5.1f},{:5.1f}), best_position = ({:5.1f},{:5.1f},{:5.1f})".format(
                i_episode, score, best_score, task._get_obs()[0][0], task._get_obs()[0][1], task._get_obs()[0][2], best_x, best_y, best_z), end="")
            break
    reward_results['episode'].append(i_episode)
    reward_results['reward'].append(score)
    sys.stdout.flush()
