# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:31:12 2019

@author: edgars
"""

import glob
import os

import create_images as ci
import only_floor as of
import pred_Surface as ps
import split_file as sf

def predict(path):
    files = glob.glob(os.path.join(path, "*.txt"))
    for file in files:
        # sadalit failu pa 50y failiem (daudzums, x, y, z, r, g, b)  un uztaisit blivak punktus
        print("1. Split file\n")

        sf.split_file(file, 10)

    # izveidot bildes no point cloud failiem
        print("2. Create images\n")

        ci.create_images(file)

    # Paredzet gridu
        print("3. Predict ground\n")

        ps.pred_Surface(file)

    #Atstat pointcloud tikai gridu
        print("4. Leave only floors\n")

        of.only_floor(file)
    # Salikt point cloud kopa
        print("5. Put together point cloud\n")

        print("Done")
