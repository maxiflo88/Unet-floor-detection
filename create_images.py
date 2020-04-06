import glob
import time
import pandas as pd
import os
import numpy as np
import cv2
import open3d as o3d
import json


def rotate(image, angle):
    h = image.shape[0]
    w = image.shape[1]

    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (h, w))


def upgradeImage(image, size=15):

    # lielaki punkti
    image = cv2.erode(image, None, iterations=size)
    # mazaki punkti
    image = cv2.dilate(image, None, iterations=size)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    return rotate(image, 270)


def pcToImage(data, maxY, maxZ, minY, minZ, image_size=256):
    # data = pd.DataFrame(data=data)
    maxY = maxY-minY
    maxZ = maxZ-minZ
    image_size1 = image_size-1
    maxY = image_size1/maxY
    maxZ = image_size1/maxZ
    img = np.ones(shape=[image_size, image_size, 3], dtype=np.uint8)*255

    database = []
    for i in range(data.shape[0]):

        x = int(round((data[i, 1]-minY)*maxY))
        y = int(round((data[i, 2]-minZ)*maxZ))
        database.append(
            [x, y, data[i, 0], data[i, 1], data[i, 2], int(data[i, 3]), int(data[i, 4]), int(data[i, 5])])

        img[x, y] = (int(data[i, 6]), int(
            data[i, 7]), int(data[i, 8]))

    return np.asarray(database), upgradeImage(img)


def createNormals(data):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.values[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data.values[:, 3:6])
    # Make smaller point cloud
    # pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.005)
    o3d.geometry.PointCloud.estimate_normals(
        pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.array(pcd.normals)
    normals[:, :3] = normals[:, :3]-np.amin(normals[:, :3], axis=0)
    normals[:, :3] = normals[:, :3]/np.amax(normals[:, :3], axis=0)
    normals[:, :3] = normals[:, :3]*255

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    points = np.concatenate((points, colors), axis=1)
    points = np.concatenate((points, normals), axis=1)
    points = pd.DataFrame(points)
    points.columns = ['x', 'y', 'z', 'r', 'g', 'b', 'nr', 'ng', 'nb']
    return points


def create_images(path, image_size=256, size=0.3):
    start_time = time.time()

    MAP = os.path.join(path.rsplit("\\", 1)[0], path.split(
        "\\")[-1].replace(".txt", ""))
    files = glob.glob(os.path.join(MAP, "*.h5"))
    count = 1
    for f in files:
        #distance, x, y, z
        hf = pd.read_hdf(f, 'df')

        maxX = hf['x'].max()
        minX = hf['x'].min()
        maxY = hf['y'].max()
        minY = hf['y'].min()
        maxZ = hf['z'].max()
        minZ = hf["z"].min()
        hf = createNormals(hf)

        range1 = minX
        range2 = minX+size
        run = True
        run2 = False
        dataset = []
        while run == True:
            data = hf.loc[(hf.x >= range1) & (hf.x < range2)]
            #pielikt ja tikai 15 punkti tad rauj nakamos metrus klat
            data = np.expand_dims(data, axis=-1)
            dataset.append(data)
            range1 = range2
            range2 += size
            if run2 == True:
                run = False
            if maxX <= range2:
                range2 = maxX
                run2 = True
            if maxX <= range2+(size/3):
                range2 = maxX
                run2 = True

        dataset = pd.DataFrame(dataset)
        # parglaba pari h5 failu sakartotu pa dalam
        # dataset.to_hdf(f, "df", mode='w')
        images = []
        database = []
        count2 = 0
        for index, row in dataset.iterrows():
            for r in row:
                r = np.squeeze(r)
                try:
                    if r.shape[1] > 1:
                        if r.shape[0] > 15:
                            data, img = pcToImage(
                                r, maxY, maxZ, minY, minZ, image_size)
                            file_name = os.path.join(
                                MAP, "img"+str(count)+"_database"+str(count2))
                            images.append(img)
                            np.save(file_name, data)
                            print("Created ", os.path.join(MAP, "img" +
                                                           str(count)+"_database"+str(count2)), "\n")
                            count2 += 1
                            # database.append(data)
                except Exception as e:
                    print("Error:", r.shape, " ", r.size)
                    print(e)
        # database = pd.DataFrame(database)
        # file_name = os.path.join(MAP, "database"+str(count))
        # np.save(file_name, database.values)
        images = np.asarray(images)
        np.save(os.path.join(MAP, "img"+str(count)), images)
        print("Created ", os.path.join(MAP, "img"+str(count)), "\n")
        count += 1

    end_time = time.time()
    print(round(end_time-start_time, 3), " seconds\n")
