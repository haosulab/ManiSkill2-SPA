import numpy as np

from ravens.utils import utils


def transform_camera_to_world(points, extrinsic):
    A = (points - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    return A


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
