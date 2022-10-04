import os.path as osp

import gym
import numpy as np
import pymp
import sapien.core as sapien
from transforms3d.euler import *

import mani_skill2.envs
from mani_skill2 import ASSET_DIR
from mani_skill2.envs.pick_and_place.pick_single import PickSingleEnv
from mani_skill2.utils.trimesh_utils import get_actor_mesh
# LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri python contact_graspnet/inference.py --np_path=test_data/065-a_cups.npy  --forward_passes=5

def main():
    np.set_printoptions(suppress=True, precision=4)

    result_path = "../contact_graspnet/results"
    import json
    with open(osp.join(osp.dirname(__file__), "PickSingleYCB-v0.train.stage1.json")) as f:
        episode_cfgs = json.load(f)
    
    episode_cfg = episode_cfgs["episodes"][32]
    env = gym.make("PickSingleYCB-v0", control_mode="pd_joint_pos", obs_mode="pointcloud_robot_seg")
    obs = env.reset(**episode_cfg["reset_kwargs"])
    model_id = episode_cfg["reset_kwargs"]["model_id"]
    seed = episode_cfg["reset_kwargs"]["seed"]
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
    def preprocess_rgbd(rgbd):
        K = np.zeros(6)
        return dict(
            rgb=rgbd['rgb'],
            depth=rgbd['depth'][:,:,0],
            K=rgbd['camera_intrinsic']
        )
    np.save(
        f"../contact_graspnet/test_data/{model_id}.npy",
        preprocess_pcd(obs['pointcloud']),
        # preprocess_rgbd(obs['image']['base_camera'])
    )
   
    import ipdb;ipdb.set_trace()
    preds = np.load(
        osp.join(result_path, f"predictions_{model_id}.npz"),
        allow_pickle=True,
    )
    predicted_poses = preds["pred_grasps_cam"].reshape(1)[0][-1]
    scores = preds["scores"].reshape(1)[0][-1]
    successes = []
    best_pose_idx = np.argmax(scores)
    pose = predicted_poses[best_pose_idx]
    result = solve(env, pose, seed=seed, model_id=model_id, debug=True, vis=True)
    # exit()
    print(f"Pose {best_pose_idx}: {pose}")
    print(result)
    # for pose in preds["pred_grasps_cam"].item()[-1]:
    #     print(f"Trying pose {pose}")
    #     success = solve(env, pose, seed=seed, model_id=model_id, debug=True, vis=True)
    #     if success is None:
    #         continue
    #     successes.append(success["success"])

    # print(model_id)
    # print(np.array(successes).astype("int").mean())

    env.close()


def solve(
    env: PickSingleEnv,
    pred_pose: np.ndarray,
    seed=None,
    model_id=None,
    debug=False,
    vis=True,
):
    # vis=False
    obs = env.reset(seed=seed, model_id=model_id)
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
            print("MP failed with the message {}".format(plan["reason"]))
            success = False
        # if not success: return success
        positions = plan["position"]
        for i in range(len(positions)):
            # print(i, len(positions))
            if vis:
                env.render()
            action = np.hstack([positions[i], gripper_action])
            env.step(action)
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

    if debug:
        import trimesh
        scene_pcd = obs['pointcloud']['xyzw'][:,:-1]
        coord_frame = trimesh.creation.axis(
            transform=grasp_pose.to_transformation_matrix(), origin_size=0.001,
            axis_radius=0.001, axis_length=0.5
        )
        trimesh.Scene([trimesh.PointCloud(scene_pcd, colors=obs['pointcloud']['rgb']), coord_frame]).show()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose.transform(sapien.Pose([0, 0.0, -0.1]))
    plan = planner.plan_screw(reach_pose, env.agent.robot.get_qpos())
    if not execute_plan(plan, OPEN_GRIPPER_POS):
        return
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
        return
    render_wait()
    plan = {
        "status": "success",
        "position": np.repeat(plan["position"][-1:], 5, axis=0),
    }
    if not execute_plan(plan, CLOSE_GRIPPER_POS):
        return
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
        return
    print("moved to goal")

    # Keep until done
    info = env.get_info()
    done = env.get_done(info=info)
    while not done:
        if vis:
            env.render()
        _, _, done, info = env.step(None)
        if done:
            break

    print(info)
    return info


if __name__ == "__main__":
    main()
