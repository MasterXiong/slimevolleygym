#!/usr/bin/env python3

# trains slime agent in selfplay from states with multiworker via MPI (fast wallclock time)
# run with
# mpirun -np 96 python train_ppo_mpi.py (replace 96 with number of CPU cores you have.)

import argparse
import os
import gym
import slimevolleygym
import numpy as np

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger, PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback

from shutil import copyfile # keep track of generations

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
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
    # load model if it's there
    modellist = [f for f in os.listdir(self.log_dir) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      if self.opponent_mode == 'latest':
        filename = os.path.join(self.log_dir, modellist[-1]) # the latest best model
      elif self.opponent_mode == 'random':
        filename = os.path.join(self.log_dir, modellist[np.random.choice(len(modellist), 1)[0]]) # randomly select previously saved models
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO1.load(filename, env=self)
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

def rollout(env, policy):
  """ play one agent vs the other in modified gym-style loop. """
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    action, _states = policy.predict(obs)
    obs, reward, done, _ = env.step(action)

    total_reward += reward

    if RENDER_MODE:
      env.render()

  return total_reward

def train():

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--opponent_mode', type=str)
  parser.add_argument('--log_dir', type=str)
  args = parser.parse_args()

  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])

  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = SlimeVolleySelfPlayEnv(args)
  env.seed(SEED)

  env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                   optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)

  eval_callback = SelfPlayCallback(env,
    best_model_save_path=args.log_dir,
    log_path=args.log_dir,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(self.log_dir, "final_model")) # probably never get to this point.

  env.close()

if __name__=="__main__":

  train()
