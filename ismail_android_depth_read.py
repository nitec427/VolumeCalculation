# from PIL import Image
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import time

from depthapp_common import *

import argparse


# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.doc(nargout=0) # opens matlab help document


def visualize_rgb_mask_depth(rgb, mask, depth_map):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rgb)
    ax2.imshow(mask)
    ax3.imshow(depth_map)
    plt.show()


def convert_depthmap_to_points(depth_map, focal_length_in_pixels=None, principal_point=None, rgb_image=None,
                               is_depth_along_z=True):
    """
    Converts given depth map to point cloud
    :param depth_map: A grayscale image where values are equal to depth in meters
    :param focal_length_in_pixels: Focal lenght of the camera measured in pixels
    :param principal_point: Center of image
    :param rgb_image: Colored image that matches depth map, used for coloring points
    :param is_depth_along_z: If True, z coordinate of the pixel will be equal to depth, otherwise depth will be
    calculated as distance between camera and pixel
    :return: List of points where each point is of form [x, y, z, r, g, b]
    """
    if focal_length_in_pixels is None:
        focal_length_in_pixels = 715
    if principal_point is None:
        principal_point = [depth_map.shape[0] / 2, depth_map.shape[1] / 2]

    if depth_map.shape[0] == 1:
        depth_map = np.swapaxes(depth_map, 0, 1).swapaxes(1, 2)

    # A point contains x,y,z,r,g,b
    points = np.ones(shape=(depth_map.shape[0] * depth_map.shape[1], 6))
    print("Points Shape: ", points.shape)
    if rgb_image is None:
        points[:, 3:6] = [0.5, 0.7, 1]
    else:
        if rgb_image.shape[0] == 3:
            rgb_image = np.swapaxes(rgb_image, 0, 1).swapaxes(1, 2)
        rgb_image = rgb_image.reshape(-1, 3)
        points[:, 3:6] = rgb_image / 256.0

    y, x = np.meshgrid(
        np.arange(0, depth_map.shape[1]), np.arange(0, depth_map.shape[0]))
    yx_coordinates = np.array([x.flatten(), y.flatten()], np.float32).T
    yx_coordinates += -1 * np.array(principal_point)
    yx_coordinates = np.flip(yx_coordinates, 1)

    points[:, 0:2] = yx_coordinates
    points[:, 2] = depth_map.flatten()

    pixel_dist = (points[:, 0] ** 2 + points[:, 1] ** 2) ** 0.5
    focal_target_dist = (focal_length_in_pixels ** 2 + pixel_dist ** 2) ** 0.5

    if not is_depth_along_z:
        points[:, 2] = points[:, 2] * \
            focal_length_in_pixels / focal_target_dist

    points[:, 0] = points[:, 0] * points[:, 2] / focal_length_in_pixels
    points[:, 1] = points[:, 1] * points[:, 2] / focal_length_in_pixels

    points[:, 1] *= -1
    return points, depth_map, rgb_image


def volume_estimation_ransac(path, id):
    start = time.time()

    # ---
    # --- 1) READ IMAGES
    # ---

    rgb, depth_map, rgb_pixel_crop, principal_point, focal_length = read_depth_map_android(path, id,
                                                                                           depth_map_resolution=(
                                                                                               90 * 2, 160 * 2))

    # ---
    # --- 2) READ MASK IMAGE
    # ---
    mask = Image.open(f"{path}/{id}_RGB_mask.jpeg")
    mask = np.array(mask)
    if mask.shape[0] > mask.shape[1]:
        mask = np.rot90(mask)
    mask = mask[rgb_pixel_crop:-rgb_pixel_crop, :]

    print("Max: ", np.max(depth_map), ", Min: ", np.min(depth_map))
    depth_map *= -1

    # ---
    # --- 3) GENERATE POINTCLOUD
    # ---
    # This function is used for generating point cloud from depth map and rgb image
    points, depth_map, rgb_image = convert_depthmap_to_points(-depth_map, focal_length, rgb_image=rgb,
                                                              principal_point=principal_point)

    # Inspect the point_cloud

    return depth_map, mask


if __name__ == '__main__':

    # path = 'H:/Dropbox/LogMeal/Algorithms/Mobile_depth_app_data'

    # filename = 1645958138081 # rice
    # filename = 1645958246901 # rice
    # filename = 1645958262493
    # filename = 1645961045038 # apple
    # filename = 1645961042086 # apple
    # filename = 1645961035126 #apple
    # filename = 1645960719300 # pizza

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",   "-dd",
                        default='Data', help="data folder")
    parser.add_argument("--test_file",   "-tf",  default=1645961042086,
                        help="test files: 1645958138081, 1645958246901, 1645958262493, 1645961045038, 1645961042086, 1645961035126, 1645960719300")
    args = parser.parse_args()

    path = args.data_dir
    filename = args.test_file

    print("Path: ", path)
    print("Filename: ", filename)

    volume_estimation_ransac(path, filename)

#     # main()
