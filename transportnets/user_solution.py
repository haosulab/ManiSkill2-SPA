import numpy as np
import sapien.core as sapien
from gym import spaces
from mani_skill2.evaluation.solution import BasePolicy
from utils.vision import perform_initial_scan
from ravens.utils import utils
from solver import MPSolver
from transporter import OriginalTransporterAgent
import pickle

class UserPolicy(BasePolicy):
    FINGER_LENGTH = 0.025
    OPEN_GRIPPER_POS = 1
    CLOSE_GRIPPER_POS = -1
    VERBOSE = 0

    def __init__(
        self, env_id: str, observation_space: spaces.Space, action_space: spaces.Space
    ) -> None:
        super().__init__(env_id, observation_space, action_space)

        if env_id != "AssemblingKits-v0":
            raise ValueError("User solution not setup to solve this environment")
        self.solver = MPSolver(
            env_name="AssemblingKits-v0",
            ee_link="panda_hand_tcp",
            joint_vel_limits=1.5,
            joint_acc_limits=1.5,
            debug=False,
            vis=True,
            gripper_type="gripper",
        )
        agent = OriginalTransporterAgent(
            "assemblyscan-transporter-1000-0", "assembly", ".", n_rotations=144
        )
        agent.load(100_000)

        self.agent = agent
        self.next_plan_stage = None
        self.plan_stage = "collect_data"
        self.plan = None
        self.env_step = 0
        self.plan_step = 0
        self.planned_grip = self.OPEN_GRIPPER_POS
        self.original_q_pos = None
        self.scanned_observtions = []
        with open("qpos_scan_sequence.pkl", "rb") as f:
            self.qpos_sequence = pickle.load(f)

    def reset(self, observations):
        self.next_plan_stage = None
        self.plan_stage = "collect_data"
        self.plan = None
        self.env_step = 0
        self.plan_step = 0
        self.planned_grip = self.OPEN_GRIPPER_POS
        self.solver.reset_planner()
        self.scanned_observtions = []
        self.original_q_pos = None

    def format_obs(self, observations):
        colors = []
        depths = []
        extrinsics = []
        intrinsics = []
        for obs in observations:
            for cam_name in obs["image"]:
                data = obs["image"][cam_name]
                colors.append(data["rgb"])
                depths.append(data["depth"][:, :, 0])
                extrinsics.append(obs["camera_param"][cam_name]["extrinsic_cv"])
                intrinsics.append(obs["camera_param"][cam_name]["intrinsic_cv"])
        agent_obs = dict(
            color=colors, depth=depths, extrinsics=extrinsics, intrinsics=intrinsics
        )
        return agent_obs

    def plan_if_fail(self, goal_pose, robot_qpos):
        if self.plan["status"] == "plan_failure":
            lowest_plan_step = np.inf
            for _ in range(10):
                tentative_plan = self.solver.planner.plan_birrt(goal_pose, robot_qpos)
                if tentative_plan["status"] == "success":
                    if len(tentative_plan["position"]) < lowest_plan_step:
                        self.plan = tentative_plan

    def act(self, observations):
        tcp = observations["extra"]["tcp_pose"]
        tcp = sapien.Pose(tcp[:3], tcp[3:])

        robot_qpos = observations["agent"]["qpos"]

        # for the first two stages of the plan, collect_data and init we handle the special actions
        # and logic which primarily collects multiple views of the board, then computes predictions for use in subsequent
        # plans
        if self.plan_stage == "collect_data":
            if self.env_step == 0:
                self.original_q_pos = observations["agent"]["qpos"]
                self.first_observation = observations
            # run a scripted policy to simply scan the environment and make multiple captures for better estimation
            if self.env_step < len(self.qpos_sequence):
                action, capture = self.qpos_sequence[self.env_step]
                if capture:
                    self.scanned_observtions.append(observations)
            elif self.env_step < len(self.qpos_sequence) + 4:
                action = self.original_q_pos
            self.env_step += 1
            if self.env_step >= len(self.qpos_sequence) + 4:
                self.plan_stage = "init"
            return action[:-1]
        elif self.plan_stage == "init":
            # predict the goal location and rotation
            act = self.agent.act(self.format_obs(self.scanned_observtions), None, None)
            self.rot_angle = utils.quatXYZW_to_eulerXYZ(act["pose1"][1])
            self.pred_obj_p = np.array([*act["pose0"][0][:2], 0.04])
            self.goal_p = np.array([*act["pose1"][0][:2], 0.04])

            # generate the grasp pose
            approaching = (0, 0, -1)
            # target_closing = self.env.tcp.pose.to_transformation_matrix()[:3, 1]
            target_closing = tcp.to_transformation_matrix()[:3, 1]

            # merge two RGBDs to make pointcloud here once
            def transform_camera_to_world(points, extrinsic):
                A = (points - extrinsic[:3, 3]) @ extrinsic[:3, :3]
                return A
            points = []
            for k in self.first_observation["image"].keys():
                cam_data = self.first_observation["image"][k]
                depth = cam_data["depth"][:, :, 0]
                intrinsic = self.first_observation["camera_param"][k]["intrinsic_cv"]
                extrinsic = self.first_observation["camera_param"][k]["extrinsic_cv"]
                xyz = utils.get_pointcloud(depth, intrinsic)
                xyz = xyz.reshape(-1, 3)
                xyz = transform_camera_to_world(xyz, extrinsic)
                points.append(xyz)
            import trimesh
            pcd = np.vstack(points)
            pcd = pcd[(pcd[:, 2] < 0.1) & (pcd[:, 2] > 0.027)]
            pcd = pcd[(pcd[:, 1] < 0.3) & (pcd[:, 1] > -0.3)]
            pcd = pcd[(pcd[:, 0] < 0.2) & (pcd[:, 0] > -0.15)]
            pcd = trimesh.PointCloud(pcd[:, :3])

            # pcd.show()
            obb = pcd.bounding_box_oriented

            grasp_info = self.solver.compute_grasp_info_by_obb(
                obb, approaching, target_closing, depth=self.FINGER_LENGTH
            )

            closing, center = grasp_info["closing"], grasp_info["center"]
            self.grasp_pose = self.solver.env.agent.build_grasp_pose(
                approaching, closing, center
            )
            # self.next_plan_stage = "reach"
            self.plan_stage = "reach"
        
        if self.plan is not None:
            # if current plan is complete, change plan stage to next one so we can generate a new plane
            if self.plan_step >= self.plan_length:
                # done with plan, move on to next stage
                self.plan_stage = self.next_plan_stage
                self.plan_step = 0
                self.plan = None

        # Plan organization and logic.
        # Generally first executes the current plan while setting the what the next plan is if this plan is done.
        if self.plan is None:
            if self.plan_stage == "reach":
                self.next_plan_stage = "grasp"
                reach_pos = self.grasp_pose.p
                reach_pos[2] = 0.05
                reach_pose = sapien.Pose(reach_pos, self.grasp_pose.q)
                self.plan = self.solver.planner.plan_screw(reach_pose, robot_qpos)
                self.plan_if_fail(reach_pose, robot_qpos)
                self.plan_length = len(self.plan["position"])
            elif self.plan_stage == "grasp":
                self.next_plan_stage = "close_gripper"
                self.plan = self.solver.planner.plan_screw(self.grasp_pose, robot_qpos)
                self.plan_length = len(self.plan["position"])
            elif self.plan_stage == "close_gripper":
                self.next_plan_stage = "hold_obj"
                self.plan_length = 5
                self.plan = 1
                self.planned_grip = self.CLOSE_GRIPPER_POS
            elif self.plan_stage == "hold_obj":
                self.next_plan_stage = "grasp_reach"
                hold_pos = self.grasp_pose.p
                hold_pos[2] = 0.05
                hold_pose = sapien.Pose(hold_pos, self.grasp_pose.q)
                self.planned_grip = self.CLOSE_GRIPPER_POS
                self.plan = self.solver.planner.plan_screw(hold_pose, robot_qpos)
                self.plan_length = len(self.plan["position"])
            elif self.plan_stage == "grasp_reach":
                self.next_plan_stage = "drop"
                self.planned_grip = self.CLOSE_GRIPPER_POS

                # determine correct goal pose and rotation angle.
                goal_pos = self.goal_p
                goal_pos[2] = 0.05
                obj_goal_pose = sapien.Pose(goal_pos, [1, 0, 0, 0])
                initial_tcp_q = tcp.q
                goal_tcp_euler_xyz = utils.quatXYZW_to_eulerXYZ(
                    [
                        initial_tcp_q[1],
                        initial_tcp_q[2],
                        initial_tcp_q[3],
                        initial_tcp_q[0],
                    ]
                )
                goal_tcp_q = utils.eulerXYZ_to_quatXYZW(
                    [
                        goal_tcp_euler_xyz[0],
                        goal_tcp_euler_xyz[1],
                        goal_tcp_euler_xyz[2] - self.rot_angle[2],
                    ]
                )
                goal_tcp_q = [
                    goal_tcp_q[3],
                    goal_tcp_q[0],
                    goal_tcp_q[1],
                    goal_tcp_q[2],
                ]

                translation = goal_pos - self.pred_obj_p

                # obj_pose = sapien.Pose(self.pred_obj_p)
                # obj_pose = tcp  # which is more accurate? Probably tcp
                tcp_goal_pose = obj_goal_pose# * obj_pose.inv() * tcp
                # import ipdb;ipdb.set_trace()
                tcp_goal_pose_p = tcp.p + translation
                tcp_goal_pose_p = tcp_goal_pose.p
                tcp_goal_pose_p[2] = 0.04
                # tcp_goal_pose = sapien.Pose(tcp_goal_pose_p, goal_tcp_q)
                tcp_goal_pose.set_p(tcp_goal_pose_p)
                tcp_goal_pose.set_q(goal_tcp_q)
                self.tcp_goal_pose = tcp_goal_pose

                self.plan = self.solver.planner.plan_screw(tcp_goal_pose, robot_qpos)
                self.plan_if_fail(tcp_goal_pose, robot_qpos)
                self.plan_length = len(self.plan["position"])
            elif self.plan_stage == "drop":
                self.next_plan_stage = "open_gripper"
                tcp_goal_pose_p = self.tcp_goal_pose.p
                tcp_goal_pose_p[2] = 0.03
                self.tcp_goal_pose.set_p(tcp_goal_pose_p)
                self.plan = self.solver.planner.plan_screw(
                    self.tcp_goal_pose, robot_qpos
                )
                self.plan_length = len(self.plan["position"])
            elif self.plan_stage == "open_gripper":
                self.next_plan_stage = "done"
                self.plan_length = 10
                self.plan = 1
                self.planned_grip = self.OPEN_GRIPPER_POS
                pass

            if self.VERBOSE > 0:
                print(f"Planned {self.plan_stage}")
        if self.VERBOSE > 0:
            print(self.plan_stage)
        if self.plan_stage == "done":
            return np.hstack([robot_qpos[:-2], self.OPEN_GRIPPER_POS])
        if self.plan_stage == "close_gripper":
            self.plan_step += 1
            return np.hstack([robot_qpos[:-2], self.CLOSE_GRIPPER_POS])
        if self.plan_stage == "open_gripper":
            self.plan_step += 1
            return np.hstack([robot_qpos[:-2], self.OPEN_GRIPPER_POS])
        if self.plan["status"] != "success":
            if self.VERBOSE > 0:
                print(self.plan["status"], self.plan.get("reason"))
            return np.hstack([robot_qpos[:-2], self.OPEN_GRIPPER_POS])
        # if self.plan["status"] == "plan_failure":
        qpos = self.plan["position"][self.plan_step]
        if self.solver.env.control_mode == "pd_joint_pos_vel":
            if "velocity" in self.plan:
                qvel = self.plan["velocity"][self.plan_step]
            else:  # in case n == 1
                qvel = np.zeros_like(qpos)
            action = action = np.hstack([qpos, qvel, self.planned_grip])
        else:
            action = np.hstack([qpos, self.planned_grip])
        self.plan_step += 1
        return action

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return "pd_joint_pos"
