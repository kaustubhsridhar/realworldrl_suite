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

"""Trains an ACME DMPO agent on a perturbed version of Cart-Pole."""

import os
from typing import Dict, Sequence

from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import dmpo, d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
import dm_env
import numpy as np
import realworldrl_suite.environments as rwrl
import sonnet as snt

# to prevent tensorflow from allocating all memory on GPU (ref: https://stackoverflow.com/a/55541385)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string('domain_name', 'humanoid', 'domain to solve')
flags.DEFINE_string('task_name', 'realworld_walk', 'task to solve')
flags.DEFINE_string('scheduler', 'cyclic_pos', # see def _generate_parameter(self) function in realworldrl_suite/environments/realworld_env.py !
                    "exo distribution \in ['constant', 'random_walk', 'drift_pos', 'drift_neg', 'cyclic_pos', 'cyclic_neg', 'uniform', 'saw_wave']") 
flags.DEFINE_string('level', 'easy', 'levels can be easy, medium, or hard')
flags.DEFINE_integer('num_episodes', 20000, 'Number of episodes to run for.')
flags.DEFINE_string('save_path', '../../data/rrl', 'outer part of the folder location for saving results')
flags.DEFINE_string('algo', 'dmpo', 'algo to use: options include d4pg and dmpo')

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
    
  return perturb_spec


def make_environment(domain_name: str,
                     task_name: str) -> dm_env.Environment:
  """Creates a RWRL suite environment."""
  perturb_spec = get_perturb_spec()
  environment = rwrl.load(
      domain_name=domain_name,
      task_name=task_name,
      # safety_spec=dict(enable=True),
      # delay_spec=dict(enable=True, actions=20),
      perturb_spec=perturb_spec,
      log_output=os.path.join(FLAGS.save_path, 'log.npz'),
      environment_kwargs=dict(
          log_safety_vars=True, log_every=1, flat_observation=True))
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Dict[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  # Create the policy network.
  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions)
  ])

  # The multiplexer transforms concatenates the observations/actions.
  multiplexer = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))

  # Create the critic network.
  critic_network = snt.Sequential([
      multiplexer,
      networks.DiscreteValuedHead(vmin, vmax, num_atoms),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }


def main(_):
  # Create folders
  print(f'domain_name = {FLAGS.domain_name} | task_name = {FLAGS.task_name} | scheduler = {FLAGS.scheduler} | level = {FLAGS.level}')
  FLAGS.save_path = f'{FLAGS.save_path}_{FLAGS.algo}'
  os.makedirs(FLAGS.save_path, exist_ok=True) # create folder at OG path
  FLAGS.save_path = f'{FLAGS.save_path}/{FLAGS.domain_name}_{FLAGS.task_name}_{FLAGS.scheduler}_{FLAGS.level}'
  os.makedirs(FLAGS.save_path, exist_ok=True) # create folder at specific path
  os.makedirs(os.path.join(FLAGS.save_path, 'replay_buffers'), exist_ok=True) # create replay_buffers folder inside above folder

  # Create an environment and grab the spec.
  environment = make_environment(
      domain_name=FLAGS.domain_name, task_name=FLAGS.task_name)
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize (online) and target networks.
  agent_networks = make_networks(environment_spec.actions)

  # Construct the agent.
  if FLAGS.algo == 'd4pg':
    agent_class = d4pg.D4PG
  elif FLAGS.algo == 'dmpo':
    agent_class = dmpo.DistributionalMPO
  agent = agent_class(  
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],  # pytype: disable=wrong-arg-types
      save_path=FLAGS.save_path,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.num_episodes, save_path=FLAGS.save_path)


if __name__ == '__main__':
  app.run(main)
