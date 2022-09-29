# coding=utf-8
# Copyright 2021 The Ravens Authors.
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

"""Ravens main training script."""

import os
import pickle
from ravens.utils import utils
from absl import app
from absl import flags
import numpy as np
from ravens import agents
from ravens import dataset
from dataset import DatasetManiskill
from ravens import tasks
from ravens.environments.environment import Environment
import tensorflow as tf

flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('assets_root', './assets/', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'hanoi', '')
flags.DEFINE_string('agent', 'transporter', '')
flags.DEFINE_integer('n_demos', 100, '')
flags.DEFINE_integer('n_steps', 40000, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')

  # Configure how much GPU to use (in Gigabytes).
  if FLAGS.gpu_limit is not None:
    mem_limit = 1024 * FLAGS.gpu_limit
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

  # Initialize environment and task.
#   env = Environment(
#       FLAGS.assets_root,
#       disp=FLAGS.disp,
#       shared_memory=FLAGS.shared_memory,
#       hz=480)
  import gym
  import mani_skill2.envs
  env = gym.make("AssemblingKits-v1", obs_mode="rgbd")
  # task = tasks.names["block-insertion"]() #tasks.names[FLAGS.task]()
  # task.mode = 'test'

  # Load test dataset.
#   ds = dataset.Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test'))
  ds = DatasetManiskill(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test'))

  # Run testing for each training run.
  for train_run in range(FLAGS.n_runs):
    name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir, n_rotations=144)

    # # Run testing every interval.
    # for train_step in range(0, FLAGS.n_steps + 1, FLAGS.interval):

    # Load trained agent.
    if FLAGS.n_steps > 0:
        agent.load(FLAGS.n_steps)

    # Run testing and save total rewards with last transition info.
    results = []
    for i in range(0, ds.n_episodes):
        print(f'Test: {i + 1}/{ds.n_episodes}')
        seed = i
        np.random.seed(seed)
        obs = env.reset(seed=i, episode_idx=i)
        
        # # import ipdb;ipdb.set_trace()
        colors=[]
        depths=[]
        extrinsics=[]
        intrinsics=[]
        for cam_name in obs['image']:
            data = obs['image'][cam_name]
            colors.append(data['rgb'])
            depths.append(data['depth'][:,:,0])
            extrinsics.append(data['camera_extrinsic'])
            intrinsics.append(data['camera_intrinsic'])
        agent_obs = dict(
            color=colors,
            depth=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics
        )
        act = agent.act(agent_obs, None, None)
        rot_angle = utils.quatXYZW_to_eulerXYZ(act['pose1'][1])
        
        print("pose0", act["pose0"])
        print("pose1", act["pose1"])
        
        print("goal_pos", obs['extra']['obj_goal_pos'])
        print("rotation", rot_angle[2])

        goal_diff = np.linalg.norm(
          obs['extra']['obj_goal_pos'][:2] - act["pose1"][0][:2]
        )
        print(f"GoalDiff {goal_diff}")
        import sapien.core as sapien
        obj_z = env.obj.pose.p[2]
        obj_q = env.obj.pose.q
        obj_angle = utils.quatXYZW_to_eulerXYZ([obj_q[1],obj_q[2], obj_q[3],obj_q[0]])[2]
        obj_angle = obj_angle + rot_angle[2]
        obj_q = utils.eulerXYZ_to_quatXYZW([0, 0, obj_angle])
        # obj_z = 0
        goal_p = [*act["pose1"][0][:2], obj_z]
        
        goal_q = [obj_q[3],obj_q[0],obj_q[1],obj_q[2]]
        from transforms3d.euler import euler2quat

        # Ground truth p and q
        # goal_p = [*env.objects_pos[env.object_id][:2], obj_z]
        # goal_q = euler2quat(*np.array([0, 0, env.objects_rot[env.object_id]]))
        env.obj.set_pose(sapien.Pose(p=goal_p, q=goal_q))
        import ipdb;ipdb.set_trace()
        while True:
          viewer = env.render()
          if viewer.window.key_down("b"):
            env.step(env.action_space.sample())
          if viewer.window.key_down("c"):
            break
            

if __name__ == '__main__':
    app.run(main)
