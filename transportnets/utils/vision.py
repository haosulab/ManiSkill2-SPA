import numpy as np

from ravens.utils import utils


def transform_camera_to_world(points, extrinsic):
    A = (points - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    return A

def perform_initial_scan(step: int, obs):
    """
    Performs an initial scan by scanning along two rows and returning the hand view camera capture. Moves closer to the board before starting to get higher quality captures.

    For the ManiSkill2 challenge this is necessary for some robotics solutions as the default camera capture is 128x128 which is not precise enough. 

    This uses pd_ee_delta_pose. The user solution uses pd_joint_pos, so in the first run first save the sequence of joint q_pos values as hardcoded actions to execute.
    """
    def strip_base_view(obs):
        del obs["image"]["base_camera"]
        return obs
    if step < 5:
        # get down lower
        return np.array([-.9, 0, 0.44, 0]), None, False
    def scan_horizontal(step):
        if step < 6:
            return np.array([0, -1, 0, 0]), strip_base_view(obs) if step == 5 else None, False
        if step < 9:
            return np.array([0, 1, 0, 0]), strip_base_view(obs) if step == 8 else None, False
        if step < 12:
            return np.array([0, 1, 0, 0]), strip_base_view(obs) if step == 11 else None, False
        if step < 15:
            return np.array([0, 1, 0, 0]), strip_base_view(obs) if step == 14 else None, False
        if step < 18:
            return np.array([0, 1, 0, 0]), strip_base_view(obs) if step == 17 else None, False
    if step < 5 + 18: return scan_horizontal(step - 5)
    if step < 5+18+3:
        return np.array([1, -1,0,0]), None, False
    if step < 5+18+6:
        return np.array([0, -1,0,0]), None, False
    if step < 5 + 18 + 6 + 18: return scan_horizontal(step - 5 - 18 - 6)
    return np.zeros(4), None, True


def get_fused_image(colors, depths, extrinsics, intrinsics, pix_size):
    image_size = colors[0].shape
    hmaps = []
    cmaps = []
    for i, (
        color,
        depth,
        extrinsic,
        intrinsic,
    ) in enumerate(zip(colors, depths, extrinsics, intrinsics)):
        xyz = utils.get_pointcloud(depth, intrinsic)
        xyz = xyz.reshape(-1, 3)
        xyz = transform_camera_to_world(xyz, extrinsic)
        # xyz = xyz[xyz[:,2] > 0]
        xyz = xyz.reshape(*image_size[:2], 3)
        bounds = np.array([[-0.16, 0.16], [-0.32, 0.32], [-1, 0.06]])
        hmap, cmap = utils.get_heightmap(xyz, color, bounds, pix_size)
        hmaps += [hmap]
        cmaps += [cmap]

    valid = np.sum(cmaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1  # these are blind spots.
    cmap = np.sum(cmaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(hmaps, axis=0)  # Max to handle occlusions.
    return cmap, hmap
