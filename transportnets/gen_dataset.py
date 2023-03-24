import argparse
import copy
import json
import os.path as osp
import time
from multiprocessing import Pool
from shutil import rmtree
import pickle
import gym
import h5py
import mani_skill2.envs
import numpy as np
import tqdm
from mani_skill2.trajectory.merge_trajectory import merge_h5
from transforms3d.euler import euler2quat
from utils.vision import perform_initial_scan
# def process_trajectory(output_h5: h5py.File):


# def worker():
def merge_h5(output_path: str, traj_paths, recompute_id=True):
    print("Merge to", output_path)

    merged_h5_file = h5py.File(output_path, "w")
    cnt = 0

    for traj_path in traj_paths:
        traj_path = str(traj_path)
        print("Merging", traj_path)
        h5_file = h5py.File(traj_path, "r")

        # Merge
        for traj_id in h5_file.keys():
            # Copy h5 data
            if recompute_id:
                new_traj_id = f"traj_{cnt}"
            else:
                new_traj_id = traj_id

            assert new_traj_id not in merged_h5_file, new_traj_id
            h5_file.copy(traj_id, merged_h5_file, new_traj_id)
            cnt += 1

        h5_file.close()
    merged_h5_file.close()


def main(args):
    env_name = "AssemblingKits-v0"

    env = gym.make(env_name, obs_mode='rgbd', control_mode="pd_joint_pos")
    keys = args.keys
    worker_id = args.worker
    output_file = osp.join("/tmp", f"{worker_id}.h5")
    output_h5 = h5py.File(output_file, "w")
    input_h5 = h5py.File(args.traj_name, "r")
    with open(args.json_name, "r") as f:
        json_file = json.load(f)

    reset_kwargs = {}
    for d in json_file["episodes"]:
        episode_id = d["episode_id"]
        r_kwargs = d["reset_kwargs"]
        reset_kwargs[episode_id] = r_kwargs

    cnt = 0
    pbar = tqdm.tqdm(position=worker_id, total=len(keys), desc=f"Worker_{worker_id}")
    for j, key in enumerate(keys):
        cur_episode_num = eval(key.split("_")[-1])
        traj = input_h5[key]
        grp = output_h5.create_group(f"traj_{cnt}")
        init_env_state = traj["env_states"][0]
        obs = env.reset(**reset_kwargs[cur_episode_num])
        env.set_state(init_env_state)

        observations = []
        done = False
        env_step = 0
        qpos_sequence = []
        with open("qpos_scan_sequence.pkl", "rb") as f:
            qpos_sequence = pickle.load(f)
        # run a scripted policy to simply scan the environment and make multiple captures for better estimation
        for i in range(len(qpos_sequence)):
            action, capture = qpos_sequence[i]
            obs, _, _, _ = env.step(action[:-1])
            if capture:
                observations.append(obs)

        # commented out code below was used to generate the initial qpos_scan_sequence.pkl file using pd_ee_delta_pos actions
        # while not done:
        #     action, kept_obs, done = perform_initial_scan(env_step, obs)
        #     env_step += 1
        #     if kept_obs is not None: observations.append(kept_obs)
        #     obs, _, _, _ = env.step(action)
        #     qpos = env.agent.robot.get_qpos()
        #     qpos_sequence.append((qpos, kept_obs is not None))
        #     # import ipdb;ipdb.set_trace()
        # with open("qpos_scan_sequence.pkl", "wb") as f:
        #     # pickle. ("qpos_scan_sequence", np.array(qpos_sequence))
        #     pickle.dump(qpos_sequence, f)
        rgbs = []
        depths = []
        cam_exts = []
        cam_ints = []
        for obs in observations:
            for cam in obs["image"].keys():
                rgb = obs["image"][cam]["rgb"]
                depth = obs["image"][cam]["depth"]
                rgbs.append(rgb)
                depths.append(depth)
                cam_ints.append(obs["camera_param"][cam]["intrinsic_cv"])
                cam_exts.append(obs["camera_param"][cam]["extrinsic_cv"])
        grp["rgbs"] = rgbs
        grp["depths"] = depths
        grp["cam_exts"] = cam_exts
        grp["cam_ints"] = cam_ints

        def convert_q(q):
            return np.array([q[1], q[2], q[3], q[0]])

        target_q = np.array([0, 0, env.objects_rot[env.object_id]])
        target_q = euler2quat(*target_q)
        target_p = env.objects_pos[env.object_id]

        action_grp = grp.create_group("action")
        action_grp["p"] = [env.obj.pose.p, target_p]
        action_grp["q"] = [convert_q(env.obj.pose.q), convert_q(target_q)]

        cnt += 1
        pbar.update(1)
    output_h5.close()
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual observations of trajectories given environment states."
    )
    # Configurations
    parser.add_argument(
        "--num-procs", default=1, type=int, help="Number of parallel processes to run"
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
        "--render",
        default=False,
        action="store_true",
        help="Render the environment while generating demonstrations",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug print"
    )
    parser.add_argument("--test-split", default=False, action="store_true")

    args = parser.parse_args()
    args.traj_name = osp.abspath(args.traj_name)
    args.output_name = osp.abspath(args.output_name)
    return args


if __name__ == "__main__":
    args = parse_args()
    with h5py.File(args.traj_name, "r") as h5_file:
        keys = sorted(h5_file.keys())

    if args.max_num_traj == -1:
        args.max_num_traj = len(keys)
    else:
        args.max_num_traj = min(len(keys), args.max_num_traj)
    args.num_procs = min(args.num_procs, args.max_num_traj)
    rng = np.random.RandomState(seed=0)
    rng.shuffle(keys)
    test_set_size = int(len(keys) * 0.4)
    if args.test_split:
        keys = keys[:test_set_size]
    else:
        keys = keys[test_set_size:]
    keys = keys[:args.max_num_traj]
    print(
        f"Test Split: {args.test_split}. Dataset Size: {len(keys)}"
    )

    # generate arguments
    args_iter = []
    N = args.num_procs
    batch_size = len(keys) // N
    for i in range(N):
        a = copy.deepcopy(args)
        a.worker = i
        a.keys = keys[i * batch_size : (i + 1) * batch_size]
        args_iter.append(a)
    # distribute the remaining keys
    for i in range(N * batch_size, len(keys)):
        args_iter[i % N].keys.append(keys[i])

    if N == 1:
        files = [main(args_iter[0])]
    else:
        with Pool(N) as p:
            files = list(p.imap(main, args_iter))
    merge_h5(args.output_name, files)
    for file in files:
        rmtree(file, ignore_errors=True)
