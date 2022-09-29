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

"""Image dataset."""

import os
import pickle

import numpy as np
from ravens import tasks
from ravens.tasks import cameras
import tensorflow as tf

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class DatasetManiskill:
  """A simple image dataset class."""

  def __init__(self, path):
    """A simple RGB-D image dataset."""
    self.path = path
    self.sample_set = []
    self.max_seed = -1
    self.n_episodes = 0

    # Track existing dataset if it exists.
    import h5py
    if "train" in path:
      # TODO remove hardcode
      path = "../../demos/AssemblingKits-v0/trajectory.pd_joint_delta_pos_rgbd_train_1000.h5"
    else:
      path = "../../demos/AssemblingKits-v0/trajectory.pd_joint_delta_pos_rgbd_test_600.h5"
    self.data=h5py.File(path)
    self.n_episodes = len(self.data.keys())
    self._cache = {}

  def set(self, episodes):
    """Limit random samples to specific fixed set."""
    self.sample_set = episodes

  def load(self, episode_id, images=True, cache=False):
    """Load data from a saved episode.

    Args:
      episode_id: the ID of the episode to be loaded.
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      episode: list of (obs, act, reward, info) tuples.
      seed: random seed used to initialize the episode.
    """
    # import ipdb;ipdb.set_trace()
    episode_id = f"traj_{episode_id}"
    poses_p = np.array(self.data[episode_id]["dict_str_p"])
    poses_q = np.array(self.data[episode_id]["dict_str_q"])
    # convert q in format wxyz to xyzw for this codebase
    def convert_q(q):
      return np.array([q[1], q[2], q[3], q[0]])
    action = dict(
        pose0=(poses_p[0], convert_q(poses_q[0])),
        pose1=(poses_p[1], convert_q(poses_q[1]))
    )
    seed=int((episode_id).split("_")[1])
    # obs = {'color': color[i], 'depth': depth[i]} if images else {}
    episode = []
    eps_colors = self.data[episode_id]['dict_str_obs']['dict_str_rgb']
    eps_depths = self.data[episode_id]['dict_str_obs']['dict_str_depth']
    for T in range(len(eps_colors)):
      colors=eps_colors[T].transpose(1,2,0)
      depths=eps_depths[T].transpose(1,2,0)
      # colors.shape, depths.shape
      obs = dict(
          color=(colors[:,:,:3], colors[:,:,3:]),
          depth=(depths[:,:,0], depths[:,:,1]),
          extrinsics=self.data[episode_id]['dict_str_obs']['dict_str_cam_extrinsics'][T],
          intrinsics=self.data[episode_id]['dict_str_obs']['dict_str_cam_intrinsics'][T]
      )
      episode.append((obs, action, 0, {}))
      # TODO - this action is only for specific case for assembling kit where ther is only ever one action
    return episode, seed

    def load_field(episode_id, field, fname):

      # Check if sample is in cache.
      if cache:
        if episode_id in self._cache:
          if field in self._cache[episode_id]:
            return self._cache[episode_id][field]
        else:
          self._cache[episode_id] = {}

      # Load sample from files.
      path = os.path.join(self.path, field)
      data = pickle.load(open(os.path.join(path, fname), 'rb'))
      if cache:
        self._cache[episode_id][field] = data
      return data

    # Get filename and random seed used to initialize episode.
    seed = None
    path = os.path.join(self.path, 'action')
    for fname in sorted(tf.io.gfile.listdir(path)):
      if f'{episode_id:06d}' in fname:
        seed = int(fname[(fname.find('-') + 1):-4])

        # Load data.
        color = load_field(episode_id, 'color', fname)
        depth = load_field(episode_id, 'depth', fname)
        action = load_field(episode_id, 'action', fname)
        reward = load_field(episode_id, 'reward', fname)
        info = load_field(episode_id, 'info', fname)

        # Reconstruct episode.
        episode = []
        for i in range(len(action)):
          obs = {'color': color[i], 'depth': depth[i]} if images else {}
          episode.append((obs, action[i], reward[i], info[i]))
        return episode, seed

  def sample(self, images=True, cache=False):
    """Uniformly sample from the dataset.

    Args:
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      sample: randomly sampled (obs, act, reward, info) tuple.
      goal: the last (obs, act, reward, info) tuple in the episode.
    """

    # Choose random episode.
    if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
      episode_id = np.random.choice(self.sample_set)
    else:
      episode_id = np.random.choice(range(self.n_episodes))
    episode, _ = self.load(episode_id, images, cache)

    # Return random observation action pair (and goal) from episode.
    i = np.random.choice(range(len(episode) - 1))
    sample, goal = episode[i], episode[-1]
    return sample, goal
