import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
import pickle
from tqdm import tqdm
import json
# isort: off
from solver import MPSolver
from ravens import agents
import tensorflow as tf
from transporter import OriginalTransporterAgent
import transforms3d
from ravens.utils import utils

class AssemblingKitsSolver(MPSolver):
    def __init__(self, model_name="assembly144-transporter-1000-0", model_n_step=10000, root_dir=".", n_rotations=144, debug=False, vis=False, **kwargs):
        super().__init__(
            env_name="AssemblingKits-v0",
            ee_link="panda_hand_tcp",
            joint_vel_limits=0.7,
            joint_acc_limits=0.5,
            debug=debug,
            vis=vis,
            gripper_type="gripper",
            **kwargs
        )
        cfg = tf.config.experimental
        gpus = cfg.list_physical_devices('GPU')
        if not gpus:
            print('No GPUs detected. Running with CPU.')
        else:
            cfg.set_visible_devices(gpus[0], 'GPU')

        np.random.seed(0)
        tf.random.set_seed(0)
        agent = OriginalTransporterAgent(model_name,"assembly", root_dir, n_rotations=n_rotations)
        agent.load(model_n_step)
        
        self.agent=agent

    def format_obs(self, obs):
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
        return agent_obs

    def solve(self, **kwargs) -> dict:
        super().solve(**kwargs)
        # import ipdb;ipdb.set_trace()
        self.env.env._obs_mode = 'rgbd'
        obs = self.env.get_obs()
        # import ipdb;ipdb.set_trace()
        self.env.env._obs_mode = 'none'
        
        act = self.agent.act(self.format_obs(obs), None, None)
        rot_angle = utils.quatXYZW_to_eulerXYZ(act['pose1'][1])
        # import ipdb;ipdb.set_trace()
        obj_z = self.env.obj.pose.p[2]
        obj_q = self.env.obj.pose.q
        obj_angle = utils.quatXYZW_to_eulerXYZ([obj_q[1],obj_q[2], obj_q[3],obj_q[0]])[2]
        obj_angle = obj_angle + rot_angle[2]
        obj_q = utils.eulerXYZ_to_quatXYZW([0, 0, obj_angle])
        # obj_z = 0
        goal_p = [*act["pose1"][0][:2], obj_z]
        # goal_p = [*self.env.objects_pos[self.env.object_id][:2], obj_z]
        goal_q = [obj_q[3],obj_q[0],obj_q[1],obj_q[2]]
        from transforms3d.euler import euler2quat
        # goal_q = euler2quat(*np.array([0, 0, self.env.objects_rot[self.env.object_id]]))
        # print(f"Goal diff: {np.linalg.norm(goal_p[:2] - self.env.objects_pos[self.env.object_id][:2])}")

        self.add_collision(self.env.kit, "kit")
        self.add_collision(self.env.obj, "obj")

        # Compute grasp pose
        approaching = (0, 0, -1)
        target_closing = self.env.tcp.pose.to_transformation_matrix()[:3, 1]
        obb = self.get_actor_obb(self.env.obj)
        grasp_info = self.compute_grasp_info_by_obb(
            obb, approaching, target_closing, depth=self.FINGER_LENGTH
        )
        
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.env.agent.build_grasp_pose(approaching, closing, center)
        self.render_wait()

        # Reach
        # print(grasp_pose.p, act["pose0"][0])
        # grasp_pose = sapien.Pose([*act["pose0"][0][:2], 0.0], grasp_pose.q)
        reach_pos = grasp_pose.p
        reach_pos[2] = 0.04
        reach_pose = sapien.Pose(reach_pos, grasp_pose.q)
        plan = self.planner.plan_screw(reach_pose, self.env.agent.robot.get_qpos())
        if plan["status"] == "plan_failure":
            lowest_plan_step = np.inf
            for _ in range(10):
                tentative_plan = self.planner.plan_birrt(
                    reach_pose, self.env.agent.robot.get_qpos()
                )
                if tentative_plan["status"] == "success":
                    if len(tentative_plan["position"]) < lowest_plan_step:
                        plan = tentative_plan
        self.execute_plan(plan, self.OPEN_GRIPPER_POS)
        self.render_wait()

        # Grasp
        self.disable_collision("obj")
        plan = self.plan_screw(grasp_pose)
        self.execute_plan(plan, self.OPEN_GRIPPER_POS)
        self.render_wait()

        # Close gripper
        self.execute_plan2(plan, self.CLOSE_GRIPPER_POS, 10)
        self.render_wait()

        hold_pos = grasp_pose.p
        hold_pos[2] = 0.035
        hold_pose = sapien.Pose(hold_pos, grasp_pose.q)
        plan = self.plan_screw(hold_pose)
        self.execute_plan(plan, self.CLOSE_GRIPPER_POS)
        self.render_wait()

        # Move to goal pose
        goal_pos = goal_p # self.env.objects_pos[self.env.object_id].copy()
        goal_pos[2] = 0.03
        # goal_rot = R.from_euler("z", [self.env.objects_rot[self.env.object_id]])
        # obj_goal_pose = sapien.Pose(goal_pos, goal_rot.as_quat()[0][[3, 0, 1, 2]])
        obj_goal_pose = sapien.Pose(goal_pos, goal_q)
        
        tcp_goal_pose = obj_goal_pose * self.env.obj.pose.inv() * self.env.tcp.pose
        plan = self.plan_screw(tcp_goal_pose)
        if plan["status"] == "plan_failure":
            lowest_plan_step = np.inf
            for _ in range(10):
                tentative_plan = self.planner.plan_birrt(
                    tcp_goal_pose, self.env.agent.robot.get_qpos()
                )
                if tentative_plan["status"] == "success":
                    if len(tentative_plan["position"]) < lowest_plan_step:
                        plan = tentative_plan
        self.execute_plan(plan, self.CLOSE_GRIPPER_POS)
        self.render_wait()

        goal_pos[2] = 0.02
        goal_rot = R.from_euler("z", [self.env.objects_rot[self.env.object_id]])
        # obj_goal_pose = sapien.Pose(goal_pos, goal_rot.as_quat()[0][[3, 0, 1, 2]])
        obj_goal_pose = sapien.Pose(goal_pos, goal_q)
        tcp_goal_pose = obj_goal_pose * self.env.obj.pose.inv() * self.env.tcp.pose
        plan = self.plan_screw(tcp_goal_pose)
        self.execute_plan(plan, self.CLOSE_GRIPPER_POS)
        self.render_wait()

        self.execute_plan2(plan, self.OPEN_GRIPPER_POS, 10)
        self.render_wait()

        action = np.r_[plan["position"][-1], self.OPEN_GRIPPER_POS]
        while not self.done:
            if self.vis:
                self.env.render()
            _, _, self.done, self.info = self.env.step(action)
            if self.done:
                break

        return self.info

def main(args):
    episode_cfg_path = args.json_name

    np.random.seed(0)
    solver = AssemblingKitsSolver(
        model_name=args.model, model_n_step=args.n_steps, root_dir=args.root_dir, n_rotations=args.n_rotations, debug=False, vis=args.render)
    

    with open(episode_cfg_path, "r") as f:
        episode_cfgs = json.load(f)["episodes"]

    results = []
    for episode_cfg in tqdm(episode_cfgs):
        reset_kwargs = episode_cfg["reset_kwargs"]
        r = solver.solve(**reset_kwargs)
        if args.verbose > 0:
            print(reset_kwargs)
            print(r)
        results.append(r)

    with open(args.output_name, "wb") as f:
        pickle.dump(results, f)
    success = np.array([r["success"] for r in results])

    pos_correct = np.array([r["pos_correct"] for r in results])

    rot_correct = np.array([r["rot_correct"] for r in results])

    ravens_success = (pos_correct & rot_correct)

    in_slot = np.array([r["in_slot"] for r in results])
    print(f"Success: {success.mean()}, Ravens Success: {ravens_success.mean()} pos correct: {pos_correct.mean()}, rot_correct: {rot_correct.mean()}, in_slot: {in_slot.mean()}")

def parse_args():
    import argparse
    import os.path as osp
    parser = argparse.ArgumentParser(
        description="Generate visual observations of trajectories given environment states.")
    # Configurations
    parser.add_argument("--render",
                        action="store_true", help="Render the solution")
    parser.add_argument("--n-steps", default=10000,
                        type=int, help="Model checkpoint to use")
    parser.add_argument("-v", "--verbose", default=0,
                        type=int, help="Verbosity. 1 = verbose, 0 = none")
    parser.add_argument("--model", required=True,
                        help="Name of the model to use. Loaded checkpoint path will be <root_dir>/checkpoints/<model>. Note root_dir is also an argument to this script")
    parser.add_argument("--json-name", required=True, type=str,
                        help="""
        Input json path, e.g. pickcube_pd_joint_delta_pos.json |
        **Json file that contains reset_kwargs is required for properly rendering demonstrations.
        This is because for environments using more than one assets, asset is different upon each environment reset,
        and asset info is only contained in the json file, not in the trajectory file.
        For environments that use a single asset with randomized dimensions, the seed info controls the specific dimension
        used in a certain trajectory, and this info is only contained in the json file.**
        """)
    parser.add_argument("--output-name",
                        type=str, help="Output of results of this evaluation script")
    parser.add_argument("--root-dir", default=".",
                        type=str, help="Working directory (where checkpoints is located)")
    parser.add_argument("--n-rotations", default=144,
                        type=int, help="Number of binned rotations to predict. Must be the same as used when training with train.py for the selected model")
    args = parser.parse_args()

    full_model_path = osp.join(args.root_dir, "checkpoints", args.model)
    if args.output_name is None:
        args.output_name = f"results_suction_{args.json_name}.pkl"
    print(
        f"Evaluating Suction Gripper with Transporter Networks on AssemblingKits. Results will be saved to {args.output_name}")
    print(
        f"Loading checkpoint {args.n_steps} of model saved at {full_model_path}")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)