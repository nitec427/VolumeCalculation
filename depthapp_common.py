import numpy as np
import cv2
from PIL import Image


'''
Example of usage:

path = 'H:/Dropbox/LogMeal/Algorithms/Mobile_depth_app_data'
id = 1645960719300
rgb, depth_map, rgb_pixel_crop, principal_point, focal_length = read_depth_map_android(path, id, depth_map_resolution=(90*2, 160*2))

'''


def read_depth_map_android(path, id, depth_map_resolution):
    """
    Android depth map is a bit different from typical depth maps, they are stored not as image but as byte stream and
    they don't align with RGB image. They also have a different resolution. This method is used for automatically
    handling operations such as adjusting principal point, scaling depth map.
    :param path: Path to the folder that contains Android output data
    :param id: Id of the save files that the method will load.
    :param depth_map_resolution: This resolution varies depending on smartphone and is needed for converting byte stream
    of depth map to image format.
    :return: rgb, depth_map, principal_point, focal_length
    """
    rgb = Image.open(f"{path}/{id}_RGB.jpeg")
    # rgb = ImageOps.mirror(rgb)

    rgb = np.array(rgb)

    print("RGB: ", rgb.shape)
    print("Depth: ", depth_map_resolution)

    focal_length, principal_point = parse_android_camera_params(
        f"{path}/{id}_camera_config.txt")

    depth_scale_factor = rgb.shape[1] / depth_map_resolution[1]
    depth_data = np.fromfile(f"{path}/{id}_dense_depth.png", dtype=np.uint16)
    H = depth_map_resolution[0]
    W = depth_map_resolution[1]

    def extractDepth(x):
        depth_confidence = (x >> 13) & 0x7
        if depth_confidence > 6:
            return 0
        return x & 0x1FFF

    # depth_data = ImageOps.mirror(depth_data)
    depth_map = -1 * np.array([extractDepth(x)
                              for x in depth_data]).reshape(H, W)

    print('ok')
    depth_map = cv2.resize(depth_map, dsize=np.array(
        depth_map.shape[:2][::-1]) * int(depth_scale_factor), interpolation=cv2.INTER_NEAREST_EXACT)
    # depth_map = cv2.resize(depth_map, dsize=np.array(depth_map.shape[:2][::-1]) * int(depth_scale_factor), interpolation = cv2.INTER_AREA)

    # Android depth map is in milimeters, by dividing with 1000 we convert it to depth map in meters.
    depth_map = depth_map / 1000.0

    rgb_pixel_crop = int((rgb.shape[0] - depth_map.shape[0]) / 2)

    rgb = rgb[rgb_pixel_crop:-rgb_pixel_crop, :, :]
    principal_point[0] -= rgb_pixel_crop

    return rgb, depth_map, rgb_pixel_crop, principal_point, focal_length


def parse_android_camera_params(path):
    """
    This method takes path to camera configs file which is output of depth estimation application and returns
    focal length and principal point
    :param path:
    :return:
    """

    principal_point = None
    focal_length = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            name, params = line.split(":")

            if name == "focal_length":
                focal_length = float(params.split(",")[0])
            elif name == "principal_point":
                principal_point = [float(x) for x in params.split(",")][::-1]

    return focal_length, principal_point
