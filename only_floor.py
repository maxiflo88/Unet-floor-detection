import time
import os
import glob
import numpy as np
import pandas as pd
import cv2
import json


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
            start=projects*depth
            for d in range(depth):
                ids[start+d] = count
                count += 1

    return ids


def ground(h5, floor, folder, id, image_size=256):


    x = 0
    y = 0
    if floor[0].size > 0:
        x = floor[0]
        y = floor[1]
        h = pd.DataFrame(h5)
        h.columns = ['px', 'py', 'x', 'y', 'z', 'r', 'g', 'b']
        groups = h.groupby(['px', 'py'])
        data = []
        for name, group in groups:
            for i in range(x.shape[0]):
                if name[0] == x[i] and name[1] == y[i]:
                    data.append(group.values[:, 2:])
                if len(data) >= 100000:
                    f = open(os.path.join(folder, "ground.txt"), 'ab')
                    for d in data:
                        np.savetxt(
                            f, d, fmt='%.7f %.7f %.7f %i %i %i')
                    f.close()
                    data = []
        if len(data) >= 0:
            f = open(os.path.join(folder, "ground.txt"), 'ab')
            for d in data:
                np.savetxt(f, d, fmt='%.7f %.7f %.7f %i %i %i')
            f.close()

    print(id, ":", h5.shape)


def only_floor(file):
    start_time = time.time()
    MAP = os.path.join(file.rsplit("\\", 1)[0], file.split(
        "\\")[-1].replace(".txt", ""))
    files = glob.glob(os.path.join(MAP, "*.npy"))

    projects = [f for f in files if f.split("\\")[-1].count("grd")]
    for project in projects:
        # (projects, depth, image_size, image_size, channels)
        p = np.load(project)

        print("Images ", p.shape)
        value = project.split("\\")[-1].replace(".npy", "")
        value = int(value.replace("grd", ""))
        # h5_name = os.path.join(MAP, "database"+str(value)+".npy")

        # hf = np.load(h5_name)
        files = glob.glob(os.path.join(MAP, "*.npy"))
        h5 = [f for f in files if f.split(
            "\\")[-1].count("img"+str(value)+"_database")]
        print("Point cloud ", len(h5))

        ids = findID(len(h5), depth=32)
        id = 0
        for depth in p:
            for d in depth:
                try:
                    # Search for floor which is 1 and everything else is 0
                    img = rotate(d, 90)
                    result = np.where(img == 1)
                    h5_name = os.path.join(
                        MAP, "img"+str(value)+"_database"+str(ids[id])+".npy")
                    hf = np.load(h5_name)

                    ground(hf, result, MAP, ids[id], 256)

                except Exception as e:
                    print("id: ", id, " ids:", ids.shape, "images:", p.shape)

                    print(e)
                id += 1

    end_time = time.time()
    print(round(end_time-start_time, 3), " seconds\n")
