import numpy as np
import cv2
import numpy as np
from scipy import stats
import glob
import pandas as pd
import open3d as o3d
import os
import copy


def rotate(image, angle):
    h = image.shape[0]
    w = image.shape[1]

    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (h, w))


def findID(shape, depth=32):
    ids = []
    projects = shape//depth
    remains = shape % depth

    if projects == 0:
        difference = depth-remains
        count = 0
        ids = np.zeros((depth), dtype=np.int)
        for d in range(shape):
            ids[count] = d
            count += 1
            if d < difference:
                ids[count] = d
                count += 1

    else:

        another_one = False
        if remains > 0:
            ids = np.zeros(
                ((projects+1)*depth), dtype=np.uint8)
            another_one = True
        else:
            ids = np.zeros(
                (projects*depth), dtype=np.uint8)
        for p in range(projects):
            value = depth*p
            for d in range(depth):
                ids[value] = value
                value += 1
        if another_one == True:
            count = shape-depth
            start = projects*depth
            for d in range(depth):
                ids[start+d] = count
                count += 1

    return ids


def grid(maxx, minx, maxy, miny, z):
    result = []
    sizex = (maxx-minx)/100
    sizey = (maxy-miny)/100
    pointx = minx
    pointy = miny
    for r in range(10000):
        result.append([pointx, pointy, z, 255, 0, 0])
        pointx += sizex
        if pointx >= maxx:
            pointx = minx
            pointy += sizey

    return np.asarray(result)


def mode(file):
    im = np.load(file)
    print(im.shape)
    allx = []
    ally = []
    for i in im:
        for i1 in i:
            # print(i1.shape)
            i1 = rotate(i1, 90)
            result = np.where(i1 == 1)
            x = stats.mode(result[0])
            y = stats.mode(result[1])
            allx.append(x[0])
            ally.append(y[0])
    allx = np.asarray(allx)
    allx = stats.mode(allx)
    allx = np.squeeze(allx[0])

    ally = np.asarray(ally)
    ally = stats.mode(ally)
    ally = np.squeeze(ally[0])
    print(file, " x:", allx, " y:", ally, "\n")
    return allx, ally


def normColors(file):
    a = pd.read_csv(file, sep=" ", usecols=[0, 1, 2], header=None, names=[
                    "x", "y", "z"], low_memory=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a.values)
    # Make smaller point cloud
    # pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.005)
    o3d.geometry.PointCloud.estimate_normals(
        pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.array(pcd.normals)
    normals[:, :3] = normals[:, :3]-np.amin(normals[:, :3], axis=0)
    normals[:, :3] = normals[:, :3]/np.amax(normals[:, :3], axis=0)
    normals[:, :3] = normals[:, :3]*255

    points = np.asarray(pcd.points)
    points = np.concatenate((points, normals), axis=1)
    np.savetxt(file, points, fmt='%.7f %.7f %.7f %i %i %i')


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def detectPlanes(path, voxel_size=0.3):
    target_down = os.path.join(path, "ground.txt")
    source_down = os.path.join(path, "ground.txt")
    source_down = pd.read_csv(source_down, sep=" ", usecols=[0, 1, 2], header=None, names=[
        "x", "y", "z"], low_memory=False)
    target_down = pd.read_csv(target_down, sep=" ", usecols=[0, 1, 2], header=None, names=[
        "x", "y", "z"], low_memory=False)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_down.values)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target_down.values)
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        pcd, pcd2, voxel_size)
    result = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source_down, target_down, result.transformation)
    # result = np.asarray(result.points)


def putTogether(h5, floor, folder, size=1):
    for h in h5:
        h = np.load(h)
        h = pd.DataFrame(h)
        h.columns = ['px', 'py', 'x', 'y', 'z', 'r', 'g', 'b']
        groups = h.groupby(['px', 'py'])
        data = []
        miny = floor-size
        maxy = floor+size
        for name, group in groups:
            if name[1] >= miny and name[1] <= maxy:
                data.append(group.values[:, 2:])
            if len(data) >= 100000:
                f = open(os.path.join(folder, "mode.txt"), 'ab')
                for d in data:
                    np.savetxt(
                        f, d, fmt='%.7f %.7f %.7f %i %i %i')
                f.close()
                data = []
        if len(data) >= 0:
            f = open(os.path.join(folder, "mode.txt"), 'ab')
            for d in data:
                np.savetxt(f, d, fmt='%.7f %.7f %.7f %i %i %i')
            f.close()


def tests(path):
    files = glob.glob(os.path.join(path, "*.txt"))
    for f in files:
        MAP = os.path.join(f.rsplit("\\", 1)[0], f.split(
            "\\")[-1].replace(".txt", ""))
        allNpy = glob.glob(os.path.join(MAP, "*npy"))
        grd = [a for a in allNpy if a.split("\\")[-1].count("grd")]
        for g in grd:
            x, y = mode(g)
            value = g.split("\\")[-1].replace(".npy", "")
            value = int(value.replace("grd", ""))
            print("img"+str(value)+"_database\n")
            data = [a for a in allNpy if a.split(
                "\\")[-1].count("img"+str(value)+"_database")]
            ids = findID(len(data))
            putTogether(data, y, path)
            normColors(os.path.join(path, "mode.txt"))


def clusterblaster(path):
    files = glob.glob(os.path.join(path, "*.txt"))
    for f in files:
        print(f, "\n")
        data = np.loadtxt(f, usecols=[0, 1, 2])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        o3d.visualization.draw_geometries([pcd])
        # labels = np.array(pcd.cluster_dbscan(
        #     eps=10, min_points=100, print_progress=True))
        # max_label = labels.max()
        # print(max_label + 1)
        # colors = cmap(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcd])
# print(all)
# pro=glob.glob(os.path.join(path,"*.txt"))
# for p in pro:
#     print(p)
#     a=pd.read_csv(p, sep=" ", usecols=[0, 1, 2], header=None, names=["x", "y", "z"], low_memory=False)
