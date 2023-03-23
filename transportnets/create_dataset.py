"""
Code adapted from maniskill2's replay trajectory code
"""
import argparse
import json
import os
import os.path as osp
from multiprocessing import Process

import h5py
import numpy as np
from transforms3d.euler import euler2quat

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from mani_skill2.trajectory.merge_trajectory import merge_h5
from maniskill2_learn.env import ReplayMemory, import_env, make_gym_env
from maniskill2_learn.utils.data import DictArray, GDict, f64_to_f32
from maniskill2_learn.utils.file import merge_h5_trajectory
from maniskill2_learn.utils.math import split_num
from maniskill2_learn.utils.meta import flush_print, get_total_memory

# from maniskill2_learn.utils.data import compress_f64


def auto_fix_wrong_name(traj):
    if isinstance(traj, GDict):
        traj = traj.memory
    for key in traj:
        if key in [
            "action",
            "reward",
            "done",
            "env_level",
            "next_env_level",
            "next_env_state",
            "env_state",
        ]:
            traj[key + "s"] = traj[key]
            del traj[key]
    return traj


tmp_folder_in_docker = "/tmp"


def render(env):
    viewer = env.render("human")


def convert_state_representation(keys, args, worker_id, main_process_id):
    input_dict = {
        "env_name": args.env_name,
        "unwrapped": True,
        "obs_mode": args.obs_mode,
        "obs_frame": args.obs_frame,
        "reward_mode": args.reward_mode,
        "control_mode": args.control_mode,
        "n_points": args.n_points,
        "n_goal_points": args.n_goal_points,
    }
    env = make_gym_env(**input_dict)
    assert hasattr(env, "get_obs"), f"env {env} does not contain get_obs"

    with open(args.json_name, "r") as f:
        json_file = json.load(f)

    reset_kwargs = {}
    for d in json_file["episodes"]:
        episode_id = d["episode_id"]
        r_kwargs = d["reset_kwargs"]
        reset_kwargs[episode_id] = r_kwargs

    cnt = 0
    output_file = osp.join(tmp_folder_in_docker, f"{worker_id}.h5")
    output_h5 = h5py.File(output_file, "w")
    input_h5 = h5py.File(args.traj_name, "r")

    for j, key in enumerate(keys):
        cur_episode_num = eval(key.split("_")[-1])
        trajectory = GDict.from_hdf5(input_h5[key])
        trajectory = auto_fix_wrong_name(trajectory)
        print("Reset kwargs for the current trajectory:", reset_kwargs[cur_episode_num])
        env.reset(**reset_kwargs[cur_episode_num])

        requires_rollout_w_actions = not "env_states" in trajectory.keys()
        if not requires_rollout_w_actions:
            length = trajectory["env_states"].shape[0] - 1
        else:
            assert "env_init_state" in trajectory.keys()
            length = trajectory["actions"].shape[0]
        assert (
            length == trajectory["actions"].shape[0] == trajectory["success"].shape[0]
        )

        replay = ReplayMemory(2)
        next_obs = None

        transport_data = []
        for i in range(length):
            if not requires_rollout_w_actions:
                if next_obs is None:
                    env_state = trajectory["env_states"][i]
                    env.set_state(env_state)
                    obs = env.get_obs()
                else:
                    obs = next_obs
                _, reward, _, _ = env.step(trajectory["actions"][i])
                # ^ We cannot directly get rewards when setting env_state.
                # Instead, reward is only accurate after env.step(); otherwise e.g. grasp criterion will be inaccurate due to zero impulse
                next_env_state = trajectory["env_states"][i + 1]
                env.set_state(next_env_state)
                next_obs = env.get_obs()
            else:
                if i == 0:
                    env.set_state(trajectory["env_init_state"])
                if next_obs is None:
                    obs = env.get_obs()
                else:
                    obs = next_obs
                next_obs, reward, _, _ = env.step(trajectory["actions"][i])

            item_i = {
                "obs": obs,
                "actions": trajectory["actions"][i],
                "dones": trajectory["success"][i],
                "episode_dones": False if i < length - 1 else True,
                "rewards": reward,
            }
            if i == 0:
                # At the first step, we simply record the observation and the object pose and target pose, which serves as our "action"
                transport_data.append(dict(obs=obs, p=env.obj.pose.p, q=env.obj.pose.q))
                item_i = GDict(transport_data[-1]).f64_to_f32()
                replay.push(item_i)
                target_q = np.array([0, 0, env.objects_rot[env.object_id]])
                target_q = euler2quat(*target_q)
                target_p = env.objects_pos[env.object_id]
                transport_data.append(dict(obs=obs, p=target_p, q=target_q))
                item_i = GDict(transport_data[-1]).f64_to_f32()
                replay.push(item_i)
                break
            if args.with_next:
                item_i["next_obs"] = next_obs

            if args.render:
                if args.debug:
                    print("reward", reward)
                render(env)

        if worker_id == 0:
            flush_print(
                f"Convert Trajectory: completed {cnt + 1} / {len(keys)}; this trajectory has length {length}"
            )
        group = output_h5.create_group(f"traj_{cnt}")
        group["seed"] = reset_kwargs[cur_episode_num]["seed"]
        group["episode_idx"] = reset_kwargs[cur_episode_num]["episode_idx"]
        cnt += 1
        # group["transport_data"] = transport_data
        replay.to_hdf5(group, with_traj_index=False)
    output_h5.close()
    flush_print(f"Finish using {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual observations of trajectories given environment states."
    )
    # Configurations
    parser.add_argument(
        "--num-procs", default=1, type=int, help="Number of parallel processes to run"
    )
    parser.add_argument(
        "--env-name", required=True, help="Environment name, e.g. PickCube-v0"
    )
    parser.add_argument(
        "--traj-name",
        required=True,
        help="Input trajectory path, e.g. pickcube_pd_joint_delta_pos.h5",
    )
    parser.add_argument(
        "--json-name",
        required=True,
        type=str,
        help="""
        Input json path, e.g. pickcube_pd_joint_delta_pos.json |
        **Json file that contains reset_kwargs is required for properly rendering demonstrations.
        This is because for environments using more than one assets, asset is different upon each environment reset,
        and asset info is only contained in the json file, not in the trajectory file.
        For environments that use a single asset with randomized dimensions, the seed info controls the specific dimension
        used in a certain trajectory, and this info is only contained in the json file.**
        """,
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output trajectory path, e.g. pickcube_pd_joint_delta_pos_pcd.h5",
    )

    parser.add_argument(
        "--max-num-traj",
        default=-1,
        type=int,
        help="Maximum number of trajectories to convert from input file",
    )
    parser.add_argument(
        "--obs-mode", default="pointcloud", type=str, help="Observation mode"
    )
    parser.add_argument(
        "--control-mode",
        default="pd_joint_delta_pos",
        type=str,
        help="Environment control Mode",
    )
    parser.add_argument(
        "--reward-mode",
        default="dense",
        type=str,
        choices=["dense", "sparse"],
        help="Reward Mode (dense / sparse)",
    )
    parser.add_argument(
        "--with-next",
        default=False,
        action="store_true",
        help="Add next_obs into the output file (for e.g. SAC+GAIL training)",
    )
    parser.add_argument(
        "--render",
        default=False,
        action="store_true",
        help="Render the environment while generating demonstrations",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug print"
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Force-regenerate the output trajectory file",
    )

    parser.add_argument("--test-split", default=False, action="store_true")

    # Specific point cloud generation args
    parser.add_argument(
        "--n-points",
        default=1200,
        type=int,
        help="If obs_mode == 'pointcloud', the number of points to downsample from the original point cloud",
    )
    parser.add_argument(
        "--n-goal-points",
        default=-1,
        type=int,
        help="If obs_mode == 'pointcloud' and 'goal_pos' is returned from environment observations (in obs['extra']), \
            then randomly sample this number of points near the goal to the returned point cloud. These points serve as helpful visual cue. -1 = disable",
    )
    parser.add_argument(
        "--obs-frame",
        default="base",
        type=str,
        choices=["base", "world", "ee", "obj"],
        help="If obs_mode == 'pointcloud', the observation frame (base/world/ee/obj) to transform the point cloud.",
    )

    args = parser.parse_args()
    args.traj_name = osp.abspath(args.traj_name)
    args.output_name = osp.abspath(args.output_name)
    print(f"Obs mode: {args.obs_mode}; Control mode: {args.control_mode}")
    if args.obs_mode == "pointcloud":
        print(
            f"Obs frame: {args.obs_frame}; n_points: {args.n_points}; n_goal_points: {args.n_goal_points}"
        )
    return args


def main():
    os.makedirs(osp.dirname(args.output_name), exist_ok=True)
    if osp.exists(args.output_name) and not args.force:
        print(
            f"Trajectory generation for {args.env_name} with output path {args.output_name} has been completed!!"
        )
        return

    with h5py.File(args.traj_name, "r") as h5_file:
        keys = sorted(h5_file.keys())
    if args.max_num_traj < 0:
        args.max_num_traj = len(keys)
    args.max_num_traj = min(len(keys), args.max_num_traj)
    args.num_procs = min(args.num_procs, args.max_num_traj)
    rng = np.random.RandomState(seed=0)
    rng.shuffle(keys)
    print(f"Found {len(keys)} trajectories")
    test_set_size = int(len(keys) * 0.4)
    if args.test_split:
        keys = keys[:test_set_size]
    else:
        keys = keys[test_set_size:]

    keys = keys[: args.max_num_traj]
    extra_args = ()

    if args.num_procs > 1:
        running_steps = split_num(len(keys), args.num_procs)[1]
        flush_print(f"Num of trajs = {len(keys)}", f"Num of process = {args.num_procs}")
        processes = []
        from copy import deepcopy

        for i, x in enumerate(running_steps):
            p = Process(
                target=convert_state_representation,
                args=(deepcopy(keys[:x]), args, i, os.getpid(), *extra_args),
            )
            keys = keys[x:]
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        running_steps = [len(keys)]
        convert_state_representation(keys, args, 0, os.getpid(), *extra_args)

    files = []
    for worker_id in range(len(running_steps)):
        tmp_h5 = osp.join(tmp_folder_in_docker, f"{worker_id}.h5")
        files.append(tmp_h5)
    from shutil import rmtree

    rmtree(args.output_name, ignore_errors=True)
    merge_h5(files, args.output_name)
    for file in files:
        rmtree(file, ignore_errors=True)
    print(f"Finish merging files to {args.output_name}")


if __name__ == "__main__":
    args = parse_args()
    main()
