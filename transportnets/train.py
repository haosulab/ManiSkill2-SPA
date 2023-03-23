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

# Code adapted for ManiSkill2

"""Ravens main training script."""

import datetime
import os

import numpy as np
import tensorflow as tf
from absl import app, flags

from dataset import DatasetManiskill
from ravens.dataset import Dataset
# from ravens import agents
from transporter import OriginalTransporterAgent

flags.DEFINE_string("train_dir", ".", "")
flags.DEFINE_string("train_data", "train_1000.h5", "")
flags.DEFINE_string("test_data", "test_600.h5", "")
flags.DEFINE_string("task", "assembly", "")
flags.DEFINE_string("agent", "transporter", "")
flags.DEFINE_float("hz", 240, "")
flags.DEFINE_integer("n_demos", 100, "")
flags.DEFINE_integer("n_steps", 100000, "")
flags.DEFINE_integer("n_runs", 1, "")
flags.DEFINE_integer("interval", 5000, "")
flags.DEFINE_integer("gpu", 0, "")
flags.DEFINE_integer("gpu_limit", None, "")
flags.DEFINE_integer("n_rotations", 36, "")

FLAGS = flags.FLAGS


def main(unused_argv):
    # Configure which GPU to use.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices("GPU")
    if not gpus:
        print("No GPUs detected. Running with CPU.")
    else:
        cfg.set_visible_devices(gpus[FLAGS.gpu], "GPU")

    # Configure how much GPU to use (in Gigabytes).
    if FLAGS.gpu_limit is not None:
        mem_limit = 1024 * FLAGS.gpu_limit
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    # Load train and test datasets.
    train_dataset = DatasetManiskill(FLAGS.train_data)
    test_dataset = DatasetManiskill(FLAGS.test_data)

    # Run training from scratch multiple times.
    for train_run in range(FLAGS.n_runs):
        name = f"{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}"

        # Set up tensorboard logger.
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(
            FLAGS.train_dir, "logs", FLAGS.agent, FLAGS.task, curr_time, "train"
        )
        writer = tf.summary.create_file_writer(log_dir)

        # Initialize agent.
        np.random.seed(train_run)
        tf.random.set_seed(train_run)

        # agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.train_dir, FLAGS.n_rotations)
        agent = OriginalTransporterAgent(
            name, FLAGS.task, FLAGS.train_dir, FLAGS.n_rotations
        )

        # Limit random sampling during training to a fixed dataset.
        max_demos = train_dataset.n_episodes
        episodes = np.random.choice(range(max_demos), FLAGS.n_demos, False)
        train_dataset.set(episodes)

        # Train agent and save snapshots.
        while agent.total_steps < FLAGS.n_steps:
            for _ in range(FLAGS.interval):
                agent.train(train_dataset, writer)
            agent.validate(test_dataset, writer)
            agent.save()


if __name__ == "__main__":
    app.run(main)
