import os.path as osp
import pickle

import gym
import numpy as np
import pymp
import sapien.core as sapien
from transforms3d.euler import *

import mani_skill2.envs
from mani_skill2 import ASSET_DIR
from mani_skill2.envs.pick_and_place.pick_single import PickSingleEnv
from mani_skill2.utils.trimesh_utils import get_actor_mesh

def main(cfg):
    result_path = cfg["predictions_path"]
    import json
    with open(cfg["episode_cfgs_json"]) as f:
        episode_cfgs = json.load(f)
    def preprocess_pcd(pcd):
        in_cam = pcd['xyzw'][:, -1] == 1
        # in_cam = in_cam & (pcd['xyzw'][:, 2] > 0.005) # crop ground
        in_cam = in_cam & (pcd['robot_seg'][:, 0] == 0)

        rgb = pcd['rgb'][in_cam]
        # transform xyz with extrinsic
        E = env.render_camera.get_extrinsic_matrix()
        xyz = E @ pcd['xyzw'][in_cam].T
        xyz = xyz.T[:,:-1]
        return dict(
            xyz=xyz,
            xyz_color=rgb
        )
    from tqdm import tqdm
    env = gym.make("PickSingleYCB-v0", control_mode="pd_joint_pos", obs_mode="pointcloud_robot_seg", enable_shadow=True)
    for episode_cfg in tqdm(episode_cfgs["episodes"]):
        obs = env.reset(**episode_cfg["reset_kwargs"])
        model_id = episode_cfg["reset_kwargs"]["model_id"]
        seed = episode_cfg["reset_kwargs"]["seed"]
        np.save(
            f"../contact_graspnet/test_data/{model_id}_{seed}.npy",
            preprocess_pcd(obs['pointcloud']),
        )
    input(f"Initial pointclouds generated. Please run contact graspnet now and store results in {result_path} and press [enter] to continue")
    data = []
    for episode_cfg in tqdm(episode_cfgs["episodes"][96:]):
        obs = env.reset(**episode_cfg["reset_kwargs"])
        model_id = episode_cfg["reset_kwargs"]["model_id"]
        seed = episode_cfg["reset_kwargs"]["seed"]
        preds = np.load(
            osp.join(result_path, f"predictions_{model_id}_{seed}.npz"),
            allow_pickle=True,
        )
        
        predicted_poses = preds["pred_grasps_cam"].reshape(1)[0][-1]
        success = False
        print(f"Loaded predictions_{model_id}_{seed} - {len(predicted_poses)} predicted grasps")
        if len(predicted_poses) == 0: 
            print("skipping")
            # 0 attempts -> no grasp / grasp is wrong
            data.append(dict(success=False, seed=seed, model_id=model_id, attempts=0, correct_grasp=False))
            continue
        scores = preds["scores"].reshape(1)[0][-1]
        score_sorted_idx = np.argsort(scores)[::-1]
        attempts = 1
        correct_grasp = False
        for i, best_pose_idx in enumerate(score_sorted_idx):
            attempts = i + 1
            pose = predicted_poses[best_pose_idx]
            info = solve(env, pose, obs, debug=False, vis=True, verbose=0)
            if info["success"]:
                correct_grasp = True
                success = True
                break
            else:
                if not info["restartable"]: 
                    # it means we found a reachable grasp, but grasp is wrong
                    correct_grasp = False
                    break
        print(f"Success: {success} - {attempts} initial attempts")
        data.append(dict(success=success, seed=seed, model_id=model_id, attempts=attempts, correct_grasp=correct_grasp))
    successes = [d["success"] for d in data]
    print(f"Success Rate: {np.mean(successes)}")
    with open("results.pkl", "wb") as f:
        pickle.dump(data, f)

def solve(
    env: PickSingleEnv,
    pred_pose: np.ndarray,
    obs,
    debug=False,
    vis=True,
    verbose=1
):
    # vis=False
    assert env.control_mode == "pd_joint_pos", env.control_mode
    control_timestep = env.control_timestep

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def render_wait():
        if not vis or not debug:
            return
        print("Press [c] to continue")
        viewer = env.render("human")
        while True:
            if viewer.window.key_down("c"):
                break
            env.render("human")

    def execute_plan(plan, gripper_action):
        success = True
        if plan["status"] != "success":
            if verbose > 0: print("MP failed with the message {}".format(plan["reason"]))
            success = False
        # if not success: return success
        positions = plan["position"]
        visualize_now = True
        for i in range(len(positions)):
            # print(i, len(positions))
            if vis and visualize_now :
                viewer = env.render("human")
                if viewer.window.key_down("c"):
                    visualize_now = False
            action = np.hstack([positions[i], gripper_action])
            env.step(action)
        if not success:
            return dict(success=False, reason=plan["reason"])
        return success

    # -------------------------------------------------------------------------- #
    # Planner
    # -------------------------------------------------------------------------- #
    joint_names = [joint.get_name() for joint in env.agent.robot.get_active_joints()]

    planner = pymp.Planner(
        urdf=str(ASSET_DIR / "descriptions/panda_v2.urdf"),
        user_joint_names=joint_names,
        srdf=str(ASSET_DIR / "descriptions/panda_v2.srdf"),
        ee_link_name="panda_hand",
        base_pose=env.agent.robot.pose,
        joint_vel_limits=0.3,
        joint_acc_limits=0.3,
        timestep=control_timestep,
    )

    OPEN_GRIPPER_POS = 1
    CLOSE_GRIPPER_POS = -1

    # -------------------------------------------------------------------------- #
    # Point cloud
    # -------------------------------------------------------------------------- #
    obj_mesh = get_actor_mesh(env.obj)
    obj_pcd = obj_mesh.sample(1000)
    planner.scene.addOctree(obj_pcd, 0.0025, name="obj")
    # The altitude of ground is set to a small negative number for clearance.
    planner.scene.addBox([1, 1, 1], [0, 0, -0.505], name="ground")

    # -------------------------------------------------------------------------- #
    # Generate grasp pose
    # -------------------------------------------------------------------------- #
    correct = np.eye(4)
    correct[:3, :3] = euler2mat(0, 0, np.pi / 2)
    
    grasp_pose =  sapien.Pose.from_transformation_matrix(
        #  @ 
        np.linalg.inv(env.render_camera.get_extrinsic_matrix()) @
        pred_pose #
        @ correct
    )
    # FIX WEIRD OFFSET?
    # grasp_pose = grasp_pose.transform(sapien.Pose([0, -0.09, 0.0]))
    # print(env.render_camera.get_intrinsic_matrix())
    # print(env.render_camera.get_extrinsic_matrix())
    # print(grasp_pose)

    # if debug:
    # import trimesh
    # scene_pcd = obs['pointcloud']['xyzw'][:,:-1]
    # coord_frame = trimesh.creation.axis(
    #     transform=grasp_pose.to_transformation_matrix(), origin_size=0.001,
    #     axis_radius=0.001, axis_length=0.5
    # )
    # trimesh.Scene([trimesh.PointCloud(scene_pcd, colors=obs['pointcloud']['rgb']), coord_frame]).show()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose.transform(sapien.Pose([0, 0.0, -0.1]))
    plan = planner.plan_screw(reach_pose, env.agent.robot.get_qpos())
    # during the first plan if we have some issue, then we can exit out and pick a new grasp
    if plan["status"] != "success":
        if verbose > 0: print("MP failed with the message {}".format(plan["reason"]))
        return dict(success=False, restartable=True)
    if not execute_plan(plan, OPEN_GRIPPER_POS):
        return dict(success=False, restartable=False)
    render_wait()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.scene.disableCollision("obj")
    plan = planner.plan_screw(grasp_pose, env.agent.robot.get_qpos())
    execute_plan(plan, OPEN_GRIPPER_POS)
    render_wait()

    # ensure distance is closed and then close gripper
    plan = {
        "status": "success",
        "position": np.repeat(plan["position"][-1:], 30, axis=0),
    }
    if not execute_plan(plan, OPEN_GRIPPER_POS):
        return dict(success=False, restartable=False)
    render_wait()
    plan = {
        "status": "success",
        "position": np.repeat(plan["position"][-1:], 5, axis=0),
    }
    if not execute_plan(plan, CLOSE_GRIPPER_POS):
        return dict(success=False, restartable=False)
    render_wait()

    # -------------------------------------------------------------------------- #
    # Move to intermediate goal
    # -------------------------------------------------------------------------- #
    # intermediate = env.goal_pos.copy()
    # intermediate[-1] += 0.2

    # goal_T = grasp_T.copy()
    # goal_T[:3, 3] = intermediate
    # goal_pose = sapien.Pose.from_transformation_matrix(goal_T)

    # plan = planner.plan_birrt(goal_pose, env.agent.robot.get_qpos())
    # execute_plan(plan, CLOSE_GRIPPER_POS)

    # -------------------------------------------------------------------------- #
    # Move to goal
    # -------------------------------------------------------------------------- #
    # NOTE(jigu): The goal position is defined by center of mass.
    offset = env.goal_pos - env.obj_pose.p
    goal_pose = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    plan = planner.plan_birrt(goal_pose, env.agent.robot.get_qpos())
    if not execute_plan(plan, CLOSE_GRIPPER_POS):
        return dict(success=False, restartable=False)

    # Keep until done
    info = env.get_info()
    done = env.get_done(info=info)
    while not done:
        if vis:
            env.render()
        _, _, done, info = env.step(None)
        if done:
            break
    info["restartable"] = False
    return info


if __name__ == "__main__":
    main(
        dict(
            predictions_path="../contact_graspnet/results",
            episode_cfgs_json=osp.join(osp.dirname(__file__), "PickSingleYCB-v0.train.stage1.json")
        )
    )