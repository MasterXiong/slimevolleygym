import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str)
args = parser.parse_args()
LOG_DIR = [args.path]


plt.figure()
for folder in LOG_DIR:
	log = pd.read_csv('/'.join([folder, 'progress.csv']))
	plt.plot(log['total_timesteps'], log['ep_reward_mean'], label=folder)
	#plt.plot(log['TimestepsSoFar'], log['EpRewMean'], label=folder)
plt.legend()
plt.savefig('train_reward.png')
plt.close()


plt.figure()
for folder in LOG_DIR:
	eval_result = np.load('/'.join([folder, 'evaluations.npz']))
	eval_mean_reward = eval_result['results'].mean(axis=1).ravel()
	plt.plot(eval_mean_reward, label=folder)
plt.legend()
plt.savefig('eval_reward.png')
plt.close()