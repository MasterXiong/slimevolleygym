#!/usr/bin/env python3

# Simple self-play PPO trainer
'''
python training_scripts/train_ppo_selfplay_vec.py --opponent_mode latest --num_env 8 --log_dir opponent_latest_vec_8
'''

import argparse
import os
import gym
import slimevolleygym
import numpy as np

from my_ppo import PPO2
#from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

from shutil import copyfile # keep track of generations

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e4)
EVAL_EPISODES = int(10)
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self

RENDER_MODE = False # set this to false if you plan on running for full 1000 trials.


class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self, args):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.policy = self
    self.best_model = None
    self.best_model_filename = None
    self.opponent_mode = args.opponent_mode
    self.log_dir = args.log_dir
  def predict(self, obs): # the policy
    if self.best_model is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.best_model.predict(obs)
      return action
  def reset(self):
    # load model from 'opponent.zip' at the beginning of every update
    self.best_model = PPO2.load(os.path.join(self.log_dir, 'opponent.zip'))
    '''
    # load model if it's there
    modellist = [f for f in os.listdir(self.log_dir) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      if self.opponent_mode == 'latest':
        filename = os.path.join(self.log_dir, modellist[-1]) # the latest best model
      elif self.opponent_mode == 'random':
        # Bug to fix: this may change the opponent at every episode
        # and the evaluation process is not run against the same opponent as the one for training
        filename = os.path.join(self.log_dir, modellist[np.random.choice(len(modellist), 1)[0]]) # randomly select previously saved models
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO2.load(filename)
    '''
    return super(SlimeVolleySelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
  # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_mean_reward = BEST_THRESHOLD
    self.generation = 0
    self.log_dir = kwargs['log_path']
  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    if result and self.best_mean_reward > BEST_THRESHOLD:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(self.log_dir, "best_model.zip")
      backup_file = os.path.join(self.log_dir, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = BEST_THRESHOLD
    return result


def train():

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--opponent_mode', type=str)
  parser.add_argument('--log_dir', type=str)
  parser.add_argument('--num_env', type=int, default=32)
  args = parser.parse_args()

  # train selfplay agent
  logger.configure(folder=args.log_dir)

  env = SlimeVolleySelfPlayEnv(args)
  #env.seed(SEED)
  vec_env = make_vec_env(SlimeVolleySelfPlayEnv, n_envs=args.num_env, seed=0, env_kwargs={'args': args}, vec_env_cls=SubprocVecEnv)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO2(MlpPolicy, vec_env, n_steps=4096, cliprange=0.2, ent_coef=0.0, noptepochs=10,
                   learning_rate=3e-4, nminibatches=64, gamma=0.99, lam=0.95, verbose=2)
  # save the randomly initialized policy so that we can load it as the initial opponent
  model.save(os.path.join(args.log_dir, "opponent.zip"))

  eval_callback = SelfPlayCallback(env,
    best_model_save_path=args.log_dir,
    log_path=args.log_dir,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(agrs.log_dir, "final_model")) # probably never get to this point.

  env.close()

if __name__=="__main__":

  train()
