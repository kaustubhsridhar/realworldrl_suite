# coding=utf-8
# Copyright 2020 The Real-World RL Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains an OpenAI Baselines PPO agent on realworldrl.

Note that OpenAI Gym is not installed with realworldrl by default.
See also github.com/openai/baselines for more information.

This example also relies on dm2gym for its gym environment wrapper.
See github.com/zuoxingdong/dm2gym for more information.
"""

import os

from absl import app
from absl import flags
from baselines import bench
from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2
from baselines.ddpg import ddpg
import dm2gym.envs.dm_suite_env as dm2gym
import realworldrl_suite.environments as rwrl

flags.DEFINE_string('domain_name', 'humanoid', 'domain to solve')
flags.DEFINE_string('task_name', 'realworld_walk', 'task to solve')
flags.DEFINE_string('scheduler', 'constant', # see def _generate_parameter(self) function in realworldrl_suite/environments/realworld_env.py !
                    "exo distribution \in ['constant', 'random_walk', 'drift_pos', 'drift_neg', 'cyclic_pos', 'cyclic_neg', 'uniform', 'saw_wave']") 
flags.DEFINE_string('level', 'easy', 'levels can be easy, medium, or hard')
flags.DEFINE_string('save_path', '../../data/rrl', 'where to save results')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_string('network', 'mlp', 'name of network architecture')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('nsteps', 20, 'number of steps per ppo rollout')
flags.DEFINE_integer('total_timesteps', 20000 * 1000, 'total steps for experiment')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_string('algo', 'ppo2', 'algo to use')

FLAGS = flags.FLAGS

def get_perturb_spec():
  # obtaining contact_friction perturb_spec from inside realworldrl_suite/environments/humanoid.py's realworld_walk() function
  perturb_spec = {
        'enable': True,
        'period': 1,
        'scheduler': FLAGS.scheduler,
    }
  if FLAGS.level == 'easy':
    perturb_spec.update(
        {'param': 'contact_friction', 'min': 0.6, 'max': 0.8, 'std': 0.02})
  elif FLAGS.level == 'medium':
    perturb_spec.update(
        {'param': 'contact_friction', 'min': 0.5, 'max': 0.9, 'std': 0.04})
  elif FLAGS.level == 'hard':
    perturb_spec.update(
        {'param': 'contact_friction', 'min': 0.4, 'max': 1.0, 'std': 0.06})

class GymEnv(dm2gym.DMSuiteEnv):
  """Wrapper that convert a realworldrl environment to a gym environment."""

  def __init__(self, env):
    """Constructor. We reuse the facilities from dm2gym."""
    self.env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': round(1. / self.env.control_timestep())
    }
    self.observation_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.observation_spec())
    self.action_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.action_spec())
    self.viewer = None


def run():
  """Runs a PPO agent on a given environment."""

  print(f'task_name = {FLAGS.task_name} | scheduler = {FLAGS.scheduler} | level = {FLAGS.level} | save_path = {FLAGS.save_path}')
  FLAGS.save_path = f'{FLAGS.save_path}_{FLAGS.algo}'
  os.makedirs(FLAGS.save_path, exist_ok=True)
  FLAGS.save_path = f'{FLAGS.save_path}/{FLAGS.domain_name}_{FLAGS.task_name}_{FLAGS.scheduler}_{FLAGS.level}'
  os.makedirs(FLAGS.save_path, exist_ok=True)
  os.makedirs(os.path.join(FLAGS.save_path, 'replay_buffers'), exist_ok=True)

  def _load_env():
    """Loads environment."""
    raw_env = rwrl.load( # see realworldrl_suite/environments/humanoid.py's realworld_walk() function for options!
        domain_name=FLAGS.domain_name,
        task_name=FLAGS.task_name,
        perturb_spec=get_perturb_spec(),
        log_output=os.path.join(FLAGS.save_path, 'log.npz'),
        environment_kwargs=dict(
            log_safety_vars=True, log_every=20, flat_observation=True))
    env = GymEnv(raw_env)
    env = bench.Monitor(env, FLAGS.save_path)
    return env

  env = dummy_vec_env.DummyVecEnv([_load_env])

  if FLAGS.algo == 'ppo2':
    ppo2.learn(
        env=env,
        network=FLAGS.network,
        lr=FLAGS.learning_rate,
        total_timesteps=FLAGS.total_timesteps,  # make sure to run enough steps
        nsteps=FLAGS.nsteps,
        gamma=FLAGS.agent_discount,
        save_path=FLAGS.save_path,
    )
  elif FLAGS.algo == 'ddpg':
    ddpg.learn(
        env=env,
        network=FLAGS.network,
        # lr=FLAGS.learning_rate,
        total_timesteps=FLAGS.total_timesteps,  # make sure to run enough steps
        # nsteps=FLAGS.nsteps,
        gamma=FLAGS.agent_discount,
        save_path=FLAGS.save_path,
    )


def main(argv):
  del argv  # Unused.
  run()


if __name__ == '__main__':
  app.run(main)
