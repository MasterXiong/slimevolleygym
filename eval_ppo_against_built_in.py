"""
evaluate the checkpoint models during PPO training process against the build-in opponent

python eval_ppo_against_built_in.py --model_path logs_latest_100M --label latest_new --interval 50 --num_episode 20 --opponent ppo
python eval_ppo_against_built_in.py --model_path logs_latest_origin_100M --label latest_origin --interval 50 --num_episode 20 --opponent ppo
python eval_ppo_against_built_in.py --model_path logs_random_100M --label random_new --interval 50 --num_episode 20 --opponent ppo
python eval_ppo_against_built_in.py --model_path logs_random_origin_100M --label random_origin --interval 50 --num_episode 20 --opponent ppo

Evaluate PPO1 policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse
import os
import time
import pickle

import slimevolleygym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class PPOPolicy:
  def __init__(self, path):
    self.model = PPO1.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action


if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO agent.')
  parser.add_argument('--model_path', help='path to stable-baselines model.',
                        type=str, default="zoo/ppo/best_model.zip")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
  parser.add_argument('--num_episode', type=int, default=10)
  parser.add_argument('--label', type=str, default='')
  parser.add_argument('--interval', type=int, default=10)
  parser.add_argument('--opponent', type=str, default='built_in')

  args = parser.parse_args()
  render_mode = args.render
  model_path = args.model_path

  env = gym.make("SlimeVolley-v0")

  if args.opponent == 'ppo':
    env.policy = PPOPolicy('zoo/ppo/best_model.zip')

  model_list = [f for f in os.listdir(model_path) if f.startswith("episode")]
  model_list.sort()
  model_list = model_list[::args.interval]

  evaluation_curve = []
  for model_name in model_list:
    print("Loading", model_name)
    policy = PPO1.load('/'.join([model_path, model_name]), env=env) # 96-core PPO1 policy

    episode_rewards, episode_lengths = evaluate_policy(policy, env, 
      n_eval_episodes=args.num_episode, deterministic=True, render=False, return_episode_rewards=True)
    episode_rewards = np.array(episode_rewards)
    print (episode_rewards.mean())
    evaluation_curve.append([episode_rewards.mean(), episode_rewards.std()])

  with open('eval_%s_against_%s.pkl' %(args.label, args.opponent), 'wb') as f:
    pickle.dump(evaluation_curve, f)

  '''
  plt.figure()
  plt.errorbar(np.arange(len(evaluation_curve)), [x[0] for x in evaluation_curve], yerr=[x[1] for x in evaluation_curve])
  plt.savefig(model_path + '/eval_against_built_in.png')
  plt.close()
  '''
