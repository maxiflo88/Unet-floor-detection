import os
import pandas as pd
import numpy as np
import time


def readMinMax(file):
    maxZ = 0
    maxY = 0
    maxX = 0
    minZ = 0
    minY = 0
    minX = 0
    first = False
    for chunk in pd.read_csv(file, sep=" ", usecols=[0, 1, 2], header=None, names=["x", "y", "z"], chunksize=100000, low_memory=False):


        maxX = max(chunk['x'].max(), maxX)
        maxY = max(chunk['y'].max(), maxY)
        maxZ = max(chunk['z'].max(), maxZ)
        if first == False:
            minX = chunk['x'].min()
            minY = chunk['y'].min()
            minZ = chunk['z'].min()
            first = True
        else:
            minX = min(chunk['x'].min(), minX)
            minY = min(chunk['y'].min(), minY)
            minZ = min(chunk['z'].min(), minZ)

    return maxX, minX, maxY, minY, maxZ, minZ

# 1.uzlabot, lai visu projektu nem pa kvadratiem pa tadam dalam lai dalas ar tiem metriem pa ko iet uz prieksu piem 10,5 
def split_file(file, size=10):
    start_time = time.time()
    print(file, "\n")
    maxX, minX, maxY, minY, maxZ, minZ = readMinMax(file)

    PROJECT = os.path.join(file.rsplit(
        "\\", 1)[0], file.split("\\")[-1].replace(".txt", ""))
    if not os.path.exists(PROJECT):
        os.mkdir(PROJECT)


#        print("Max:",maxX," ",maxY," ", maxZ,"\n")
#        print("Min:",minX," ",minY," ", minZ,"\n")

        for chunk in pd.read_csv(file, sep=" ", usecols=[0, 1, 2, 3, 4, 5], header=None, names=["x", "y", "z", "r", "g", "b"], chunksize=100000, low_memory=False):
            range1 = minY
            range2 = minY+size
            count = 1
            run = True
            run2 = False
            while run == True:
                part = count
                part = file.split(
                    "\\")[-1].replace(".txt", "_part"+str(part)+".h5")
                file_name = os.path.join(PROJECT, part)
                data = chunk.loc[(chunk.y >= range1) & (chunk.y < range2)]
                if not os.path.isfile(file_name):
                    print("Created ", file_name, "\n")
                    data.to_hdf(file_name, "df", mode='w', append=True)
#                    data.to_csv(file_name, header='column_names')

                else:  # else it exists so append without writing the header
                    data.to_hdf(file_name, "df", mode='r+', append=True)
#                    data.to_csv(file_name, mode='a', header=False)

                range1 = range2
                range2 += size
                if run2 == True:
                    run = False
                if maxY <= range2:
                    range2 = maxY
                    run2 = True
                if maxY <= range2+(size/3):
                    range2 = maxY
                    run2 = True
                count += 1
        end_time = time.time()
        print(round(end_time-start_time, 3), " seconds\n")
