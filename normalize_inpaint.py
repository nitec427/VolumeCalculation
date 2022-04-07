from ismail_android_depth_read import volume_estimation_ransac
import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse


def normalize_img(depth_map):
    """
        Description: Gets normalized depth image and subtracts minimum value from all over the image

        :param depth_map An image that is to be normalized
        :returns multiplier that is used to scale in range [0,255]
        :returns normalized depth map
    """
    depth_map = depth_map - np.min(depth_map)
    max_val = np.max(depth_map)
    multiplier = 255 / max_val
    # Image is normalized at this step
    depth_map_normal = np.array(depth_map * multiplier, dtype=np.uint8)
    return depth_map_normal, multiplier


def inpaint_img(normal_depth, mask_img):
    """
        Description: Inpaint the normalized depth image by using mask and output inpainted img and object img that is obtained after inpaint operation

        :param depth_map        An image that is to be inpainted
        :param mask_img         Helper image that decides px location that are to be colored as black
        :returns inpainted_img  Inpainted result 
        :returns object_img 
    """

    depth_copy = normal_depth.copy()  # Do not touch normal depth

    # Remove the pxs that you want to inpaint from img
    depth_copy[mask_img == 255] = 0
    inpainted_img = cv2.inpaint(depth_copy, mask_img, 3, cv2.INPAINT_TELEA)
    object_img = np.array(inpainted_img.astype(
        np.int16) - depth_copy.astype(np.int16))
    object_img[object_img < 0] = 0

    return inpainted_img, object_img


def make_plot(axes, img, fig, title):
    """ 
        Description: Helper function for matplotlib
    """
    pos = axes.imshow(img)
    axes.title.set_text(title)
    fig.colorbar(pos, ax=axes, shrink=.75, orientation='horizontal', pad=.1)


def visualize_result(normalized_img, inpainted_img, obj, mask):
    """ 
        Description: Visualize the results by using matplotlib, no return value

        :params normalized_img Normalized version of the given depth image
        :params inpainted_img  Inpainted version of the given depth image (pxs carrying object info is blacked out with mask)
        :params obj            Object image that is obtained from inpainted image
        :params mask           Mask of the current file
    """
    figure, axes = plt.subplots(2, 2)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    make_plot(ax1, normalized_img, figure, "Normalized Image")
    make_plot(ax2, inpainted_img, figure, 'Inpainted Image')
    make_plot(ax3, obj, figure, 'Object Image')
    make_plot(ax4, mask, figure, 'Mask Image')
    plt.show()


def main_code(path, filename):
    depth, mask = volume_estimation_ransac(path, filename)
    normalized, _ = normalize_img(depth)
    inpainted, result = inpaint_img(normalized, mask)
    visualize_result(normalized, inpainted, result, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Obtain object image by using inpainting")
    parser.add_argument('-dp', '--data_path', default='Data',
                        help="Path that contain the images")
    parser.add_argument('-fi', '--file_id',
                        default='1645958138081', help="Id number of the image")
    parser.add_argument('-a', '--print_all', default='n',
                        help="Find all results")

    args = parser.parse_args()
    path = args.data_path
    filename = args.file_id

    # If all_flag is given then all of the filenames will be used.
    all_flag = True if args.print_all == 'y' else False
    filenames = [1645958138081, 1645958246901, 1645958262493, 1645961045038, 1645961042086, 1645961035126,
                 1645960719300]

    if all_flag:
        for filename in filenames:
            main_code(path, filename)

    else:
        main_code(path, filename)
        # Give the data path and the id of the image (Given Func)
