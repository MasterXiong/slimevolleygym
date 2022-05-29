import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pickle

'''
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str)
args = parser.parse_args()
LOG_DIR = [args.path]
'''
'''
plt.figure()
for folder in LOG_DIR:
	log = pd.read_csv('/'.join([folder, 'progress.csv']))
	try:
		plt.plot(log['total_timesteps'], log['ep_reward_mean'], label=folder)
	except:
		plt.plot(log['TimestepsSoFar'], log['EpRewMean'], label=folder)
plt.xlabel('training time steps')
plt.ylabel('reward mean')
plt.savefig(folder + '/reward_train.png')
plt.close()
'''
'''
plt.figure()
for folder in LOG_DIR:
	eval_result = np.load('/'.join([folder, 'evaluations.npz']))
	eval_mean_reward = eval_result['results'].mean(axis=1).ravel()
	plt.plot(eval_mean_reward, label=folder)
plt.xlabel('evaluation round')
plt.ylabel('reward mean')
plt.savefig(folder + '/reward_eval.png')
plt.close()
'''

labels = ['random_origin', 'random_new', 'latest_origin', 'latest_new']
opponent = 'ppo'
plt.figure()
for label in labels:
	with open('eval_%s_against_%s.pkl' %(label, opponent), 'rb') as f:
		evaluation_curve = pickle.load(f)
	plt.plot([x[0] for x in evaluation_curve], label=label)
plt.legend()
plt.savefig('eval_against_%s.png' %(opponent))
plt.close()
