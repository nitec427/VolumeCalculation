import argparse
from functools import reduce
from ismail_android_depth_read import volume_estimation_ransac, convert_depthmap_to_points
from normalize_inpaint import normalize_img, inpaint_img
import open3d as o3d
import numpy as np
import os
from scipy.spatial import Delaunay


def obtain_depth_data(path, filename):
    """
        :param     path     = Folder where files are kept
        :param     filename = The id of the specific file

        :returns   depth     = original depth image
        :returns   mask      = mask image for the given file
        :returns   inpainted = inpainted version of original depth image
    """
    depth, mask = volume_estimation_ransac(path, filename)
    normalized, coef = normalize_img(depth)
    inpainted, _ = inpaint_img(normalized, mask)

    inpainted = inpainted / coef
    inpainted += np.min(depth)

    return depth, mask, inpainted


def obtain_point_cloud(depth, inpainted, mask):
    """
        Description:Compute point cloud by using original depth and inpainted images

        :param     depth     = Original depth map
        :param     inpainted = The inpainted depth version (object is blacked out for this purpose)
        :param     mask      = The mask of the given file

        :returns   pcd       = The output is point cloud 
    """
    points, _, _ = convert_depthmap_to_points(inpainted)
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(points)
    # pcd2.paint_uniform_color([0,0,1])

    points2, _, _ = convert_depthmap_to_points(depth)
    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(points)
    # pcd3.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([pcd3, pcd2])
    org_depth_3d = points2[:, :3]
    inp_depth_3d = points[:, :3]
    mask_filter = np.reshape(mask, -1)
    points3 = np.concatenate(
        (np.asarray(org_depth_3d[mask_filter > 250]), np.asarray(inp_depth_3d[mask_filter > 250])), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3)
    return pcd


def compute_volume(pcd):
    """
        Description:Compute volume by using point cloud

        :param     pcd    = Given point cloud
        :returns   volume = The output is volume in m^3
    """
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # downpcd = pcd
    downpcd.paint_uniform_color([1, 0, 0])
    xyz = np.asarray(downpcd.points)
    xy_catalog = []
    for point in xyz:
        xy_catalog.append([point[0], point[1]])
    tri = Delaunay(np.array(xy_catalog))
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(xyz)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
    surface.paint_uniform_color([0, 0, 1])
    if os.getenv('DEBUG') == 'True':
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([downpcd])
        o3d.visualization.draw_geometries([surface], mesh_show_wireframe=True)
    volume = reduce(lambda a, b:  a + volume_under_triangle(b),
                    get_triangles_vertices(surface.triangles, surface.vertices), 0)
    return volume


def main_code(path, filename):
    """
        Description: Fetches the path to the desired file to the functions that obtain depth map and point cloud.

        :param     path     = Folder where files are kept
        :param     filename = The id of the specific file
    """
    depth, mask, inpainted_img = obtain_depth_data(path, filename)
    point_cloud = obtain_point_cloud(depth, inpainted_img, mask)
    volume = compute_volume(point_cloud)

    M3TOCM3 = 1000000  # 1m^3 = 10^6cm3
    print(f"The volume under the given food is {round(volume * M3TOCM3)} cm3")


def get_triangles_vertices(triangles, vertices):
    """ 
    This function is taken from this medium post
    https://jose-llorens-ripolles.medium.com/stockpile-volume-with-open3d-fa9d32099b6f

    Triangles in Open3D meshes define the indexes of the vertices, not the vertices, so we need to translate the triangles so that their values are actually 3D points and not indices to the vertices list
    """
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [vertices[triangle[0]],
                                  vertices[triangle[1]], vertices[triangle[2]]]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)


def volume_under_triangle(triangle):
    """ 
    This function is taken from this medium post
    https://jose-llorens-ripolles.medium.com/stockpile-volume-with-open3d-fa9d32099b6f

    Function to compute the volume under each triangle
    """
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)


if __name__ == '__main__':
    # Parser arguments
    parser = argparse.ArgumentParser(
        description="Obtain object image by using inpainting")
    parser.add_argument('-dp', '--data_path', default='Data',
                        help="Path that contain the images")
    parser.add_argument('-fi', '--file_id',
                        default='1645958138081', help="Id number of the image")
    parser.add_argument('-d', '--debug',
                        default='n', help="Set environment variable to DEBUG")
    parser.add_argument('-a', '--print_all', default='n',
                        help="Find all results")

    filenames = [1645958138081, 1645958246901, 1645958262493, 1645961045038, 1645961042086, 1645961035126,
                 1645960719300]

    args = parser.parse_args()

    path = args.data_path
    filename = args.file_id
    debug_mode = args.debug.lower()

    # Boolean flag that either prints all results or only prints one
    all_flag = True if args.print_all == 'y' else False

    # Configure DEBUG version of the code. If DEBUG is enabled then all visualizations are printed
    os.environ["DEBUG"] = "True" if debug_mode == 'y' else ""
    if all_flag:
        for filename in filenames:
            main_code(path, filename)
    else:
        main_code(path, filename)
