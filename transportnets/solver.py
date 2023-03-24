import gym
import numpy as np
import pymp
import sapien.core as sapien
import trimesh
from mani_skill2 import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill2.utils.common import normalize_vector
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from mani_skill2.utils.wrappers import RecordEpisode


class MPSolver:
    OPEN_GRIPPER_POS = -1
    CLOSE_GRIPPER_POS = 1
    FINGER_LENGTH = 0.025

    def __init__(
        self,
        env_name: str,
        ee_link: str,
        joint_vel_limits: float,
        joint_acc_limits: float,
        obs_mode=None,
        debug=False,
        vis=False,
        record=False,
        gripper_type="suction",
        **kwargs
    ):
        obs_mode = "none" if obs_mode is None else obs_mode
        self.env = gym.make(env_name, obs_mode=obs_mode, control_mode="pd_joint_pos")
        self.gripper_type = gripper_type
        if self.gripper_type == "suction":
            self.OPEN_GRIPPER_POS = -1
            self.CLOSE_GRIPPER_POS = 1
        else:
            self.OPEN_GRIPPER_POS = 1
            self.CLOSE_GRIPPER_POS = -1
        self.record = record
        if self.record:
            self.env = RecordEpisode(
                self.env,
                kwargs["record_dir"],
                save_trajectory=(not kwargs["no_traj"]),
                trajectory_name="trajectory",
                save_video=(not kwargs["no_video"]),
                save_on_reset=False,
                render_mode=kwargs["render_mode"],
            )

        self.ee_link = ee_link
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.urdf_path = str(self.env.agent.urdf_path).format(
            PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR
        )
        self.srdf_path = self.urdf_path.replace("urdf", "srdf")
        self.joint_names = [
            joint.get_name() for joint in self.env.agent.robot.get_active_joints()
        ]

        self.debug = debug
        self.vis = vis
        self.done = False
        self.info = {}
        self.reset_planner()

    def reset_planner(self):
        self.planner = pymp.Planner(
            urdf=self.urdf_path,
            # srdf=self.srdf_path,
            user_joint_names=self.joint_names,
            ee_link_name=self.ee_link,
            base_pose=self.env.agent.robot.pose,
            joint_vel_limits=self.joint_vel_limits,
            joint_acc_limits=self.joint_acc_limits,
            timestep=self.env.control_timestep,
        )
        self.planner.scene.addBox([1, 1, 1], [0, 0, -0.505], name="ground")

    def solve(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.reset_planner()
        self.done = False
        self.info = {}
        return obs
    def add_collision(self, actor: sapien.Actor, name: str = None):
        self.planner.scene.addOctree(
            points=get_actor_mesh(actor).sample(1500), resolution=0.0025, name=name
        )

    def add_collision_by_name(self, actor_name: str):
        self.add_collision(get_entity_by_name(actor_name), name=actor_name)

    def disable_collision(self, name: str):
        self.planner.scene.disableCollision(name)

    def remove_collision(self, name: str):
        self.planner.scene.removeGeometry(name)

    def plan_screw(
        self, target_pose: sapien.Pose, robot_qpos: np.ndarray = None, **kwargs
    ) -> dict:
        if robot_qpos is None:
            robot_qpos = self.env.agent.robot.get_qpos()
        return self.planner.plan_screw(target_pose, robot_qpos, **kwargs)

    def plan_birrt(
        self, target_pose: sapien.Pose, robot_qpos: np.ndarray = None, **kwargs
    ) -> dict:
        if robot_qpos is None:
            robot_qpos = self.env.agent.robot.get_qpos()
        return self.planner.plan_birrt(target_pose, robot_qpos, **kwargs)

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.env.render("human")
        while True:
            if viewer.window.key_down("c"):
                break
            self.env.render("human")

    def execute_plan(self, plan: dict, gripper_action: int):
        """Arm and gripper action."""
        if plan["status"] != "success":
            print(plan["status"], plan.get("reason"))
            return
        n = len(plan["position"])
        for i in range(n):
            qpos = plan["position"][i]
            if self.env.control_mode == "pd_joint_pos_vel":
                if "velocity" in plan:
                    qvel = plan["velocity"][i]
                else:  # in case n == 1
                    qvel = np.zeros_like(qpos)
                action = action = np.hstack([qpos, qvel, gripper_action])
            else:
                action = np.hstack([qpos, gripper_action])
            _, _, done, info = self.env.step(action)
            self.done = done
            self.info = info
            if self.vis:
                self.env.render()

    def execute_plan2(self, plan: dict, gripper_action: int, t: int):
        """Gripper action at the last step of arm plan."""
        if plan is None or plan["status"] != "success":
            if self.gripper_type == "suction":
                qpos = self.env.agent.robot.get_qpos()[:]
            else:
                qpos = self.env.agent.robot.get_qpos()[:-2]  # hardcode
        else:
            qpos = plan["position"][-1]
        if self.env.control_mode == "pd_joint_pos_vel":
            action = np.r_[qpos, np.zeros_like(qpos), gripper_action]
        else:
            action = np.r_[qpos, gripper_action]
        for _ in range(t):
            try:
                _, _, done, info = self.env.step(action)
            except:
                import ipdb

                ipdb.set_trace()
            self.done = done
            self.info = info
            if self.vis:
                self.env.render()

    def get_ground_pcd(self, altitude, xrange, yrange, dx, dy):
        x = np.arange(*xrange, dx)
        y = np.arange(*yrange, dy)
        xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
        xy = xy.reshape(-1, 2)
        xyz = np.concatenate([xy, np.full([xy.shape[0], 1], altitude)], axis=1)
        return xyz

    def get_actor_obb(self, actor: sapien.Actor, to_world_frame=True, vis=False):
        mesh = get_actor_mesh(actor, to_world_frame=to_world_frame)
        assert mesh is not None, "can not get actor mesh for {}".format(actor)

        obb: trimesh.primitives.Box = mesh.bounding_box_oriented

        if vis:
            obb.visual.vertex_colors = (255, 0, 0, 10)
            trimesh.Scene([mesh, obb]).show()

        return obb

    def compute_grasp_info_by_obb(
        self,
        obb: trimesh.primitives.Box,
        approaching=(0, 0, -1),
        target_closing=None,
        depth=0.0,
        ortho=True,
    ):
        """Compute grasp info given an oriented bounding box.
        The grasp info includes axes to define grasp frame,
        namely approaching, closing, orthogonal directions and center.

        Args:
            obb: oriented bounding box to grasp
            approaching: direction to approach the object
            target_closing: target closing direction,
                used to select one of multiple solutions
            depth: displacement from hand to tcp along the approaching vector.
                Usually finger length.
            ortho: whether to orthogonalize closing  w.r.t. approaching.
        """

        extents = np.array(obb.primitive.extents)
        T = np.array(obb.primitive.transform)

        # Assume normalized
        approaching = np.array(approaching)

        # Find the axis closest to approaching vector
        angles = approaching @ T[:3, :3]  # [3]
        inds0 = np.argsort(np.abs(angles))
        ind0 = inds0[-1]

        # Find the shorter axis as closing vector
        inds1 = np.argsort(extents[inds0[0:-1]])
        ind1 = inds0[0:-1][inds1[0]]
        ind2 = inds0[0:-1][inds1[1]]

        # If sizes are close, choose the one closest to the target closing
        if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
            vec1 = T[:3, ind1]
            vec2 = T[:3, ind2]
            if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
                ind1 = inds0[0:-1][inds1[1]]
                ind2 = inds0[0:-1][inds1[0]]
        closing = T[:3, ind1]

        # Flip if far from target
        if target_closing is not None and target_closing @ closing < 0:
            closing = -closing

        # Reorder extents
        extents = extents[[ind0, ind1, ind2]]

        # Find the origin on the surface
        center = T[:3, 3].copy()
        half_size = extents[0] * 0.5
        center = center + approaching * (-half_size + min(depth, half_size))

        if ortho:
            closing = closing - (approaching @ closing) * approaching
            closing = normalize_vector(closing)

        grasp_info = dict(
            approaching=approaching, closing=closing, center=center, extents=extents
        )
        return grasp_info

    def close(self):
        self.env.close()
