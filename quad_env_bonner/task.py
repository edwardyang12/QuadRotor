import sys
import pandas as pd
from quad_env.envs.agent import DDGP
import quad_env
import csv
import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make('quadcopter-v4')

num_episodes = 500
agent = DDGP(env)
best_score = -1000
best_x = 0
best_y = 0
best_z = 0
data = {}
reward_log = "reward.txt"
reward_labels = ['episode', 'reward']
reward_results = {x : [] for x in reward_labels}
episode =[]
scoreList = []
best_scoreList = []

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()
    score = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.step(action, reward, next_state, done)
        env.render()
        state = next_state
        score = reward # * (env.time * 50)
        # score = reward / (50 * env.runtime)
        # score += reward
        # score = score / (env.time * 50)
        if score > best_score:
            best_x = env.pose[0]
            best_y = env.pose[1]
            best_z = env.pose[2]
        best_score = max(score, best_score)
        data[i_episode] = {'Episode': i_episode, 'Reward':score,'Action':action,'Best_Score':best_score,
                            'Position_x':env.pose[0],'Position_y':env.pose[1],'Position_z':env.pose[2]}
        if done:
            print("\rEpisode = {:4f}, score = {:7.3f} (best = {:7.3f}), last_position = ({:5.1f},{:5.1f},{:5.1f}), best_position = ({:5.1f},{:5.1f},{:5.1f})".format(
                i_episode, score, best_score, env.pose[0], env.pose[1], env.pose[2], best_x, best_y, best_z), end="")
            episode.append(i_episode)
            scoreList.append(score)
            best_scoreList.append(best_score)
            break
    reward_results['episode'].append(i_episode)
    reward_results['reward'].append(score)
    sys.stdout.flush()

graph = plt.figure()
ax = graph.add_subplot(111)

ax.scatter(episode, scoreList, label = 'score', c='b')
ax.scatter(episode, best_scoreList, label = 'best_score', c='r')

plt.legend(loc='lower right')
plt.show()
