
import sys
import predict as prd
import os
import test
BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
#path = os.path.join(BASE_DIR, "project")
path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path = os.path.join(path, "project")
path=os.path.join(path, "eee")


print("suka")
# test.clusterblaster(path)
# test.tests(path)
# test.detectPlanes(path)
# prd.predict(path)
