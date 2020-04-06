import numpy as np
import time
import os
import glob

import tensorflow as tf

from keras.models import Model, load_model


def createTestNPY2(file, depth=32, image_size=256):
    X_train = []
    projects = file.shape[0]//depth
    remains = file.shape[0] % depth
    if projects == 0:
        difference = depth-remains
        X_train = np.zeros(
            (1, depth, image_size, image_size, 3), dtype=np.uint8)
        count = 0
        for d in range(len(file)):
            X_train[0][count] = file[d]
            count += 1
            if d < difference:
                X_train[0][count] = file[d]
                count += 1

        return X_train

    else:
        difference = depth-remains
        another_one=False
        if remains > 0:
            X_train = np.zeros(
                (projects+1, depth, image_size, image_size, 3), dtype=np.uint8)
            another_one = True
        else:
            X_train = np.zeros(
                (projects, depth, image_size, image_size, 3), dtype=np.uint8)
        for p in range(projects):
            value = depth*p
            for d in range(depth):
                X_train[p][d] = file[value]
                value += 1
        if another_one == True:
            count = file.shape[0]-depth
            for d in range(depth):
                X_train[projects][d] = file[count]
                count += 1

        return X_train


def createTestNPY(file, depth=32, image_size=256):
    X_train = []
    projects = file.shape[0]//depth
    remains = file.shape[0] % depth
    if projects == 0:
        difference = depth-remains
        X_train = np.zeros(
            (1, depth, image_size, image_size, 3), dtype=np.uint8)
        count = 0
        for d in range(len(file)):
            X_train[0][count] = file[d]
            count += 1
            if d < difference:
                X_train[0][count] = file[d]
                count += 1

        return X_train

    else:
        difference = depth-remains
        another_one = False
        if (remains+(depth/3)) >= depth:
            X_train = np.zeros(
                (projects+1, depth, image_size, image_size, 3), dtype=np.uint8)
            another_one = True
        else:
            X_train = np.zeros((projects, depth, image_size,
                                image_size, 3), dtype=np.uint8)
        for p in range(projects):
            value = depth*p
            for d in range(depth):
                X_train[p][d] = file[value]
                value += 1
        if another_one == True:
            count = 0
            rem = depth*projects
            for d in range(remains):
                X_train[projects][count] = file[rem]
                count += 1
                if d < difference:
                    X_train[projects][count] = file[rem]
                    count += 1
                rem += 1

        return X_train


def pred_Surface(file):
    start_time = time.time()

    MAP = os.path.join(file.rsplit("\\", 1)[0], file.split(
        "\\")[-1].replace(".txt", ""))
    files = glob.glob(os.path.join(MAP, "*.npy"))
    files = [f for f in files if f.split("\\")[-1].count("img")]
    files = [f for f in files if not f.split("\\")[-1].count("database")]
    model = load_model('model4.h5')
    for f, count in zip(files, range(len(files))):
        print("Load ", f, "\n")
        img = np.load(f)
        images = createTestNPY2(img, 32, img.shape[1])
        print(images.shape, " ", img.shape)
        if images.shape[0]<=4:
            pred = model.predict(images, verbose=1)
            preds = (pred > 0.8).astype(np.uint8)
            np.save(os.path.join(MAP, "grd"+str(count+1)), preds)
        else:
            together=[]
            images2 = np.zeros((1,images.shape[1], images.shape[2], images.shape[3], 3), dtype=np.uint8)
            for i in images:

                images2[0]=i
                pred = model.predict(images2, verbose=1)
                preds = (pred > 0.8).astype(np.uint8)
                together.append(preds[0])
            together=np.asarray(together)
            print(os.path.join(MAP, "grd"+str(count+1))," ",together.shape,"\n")
            np.save(os.path.join(MAP, "grd"+str(count+1)), together)

        count += 1

    end_time = time.time()
    print(round(end_time-start_time, 3), " seconds\n")
